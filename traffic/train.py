import yaml
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import time
import random
import argparse
import sys

parser = argparse.ArgumentParser(description='Federated Learning')
parser.add_argument('--setting', default='fedavg', help="Training setting (0|1|combined|fedavg), where the number represents the party")
parser.add_argument('--local-epochs', default=1, type=int, help="FedAvg per how many epochs")
parser.add_argument('--spdz', action='store_true', default=False, help='Enable SPDZ')

parser.add_argument('--batch-size', default=64, type=int, help="How many records per batch")  # same as dcrnn_xx.yaml
parser.add_argument('-E', '--epochs', default=100, type=int, help="Number of epochs")  # same as dcrnn_xx.yaml

parser.add_argument('--dp', action='store_true', default=False, help='Enable DP')
parser.add_argument('-e', '--epsilon', default=1.0, type=float, help="Privacy Budget for each party")
parser.add_argument('--lotsize-scaler', default=1.0, type=float, help="Scale the lot size sqrt(N) by a multiplier")
parser.add_argument('--clip', default=1.0, type=float, help="L2 bound for the gradient clip")

parser.add_argument('--which-gpu', default="1", help="Use which gpu")

arglist = parser.parse_known_args()
args = arglist[0]

os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpu

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib import utils
from lib.utils import load_graph_data
from dcrnn_model import DCRNNModel

from scipy.optimize import newton
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
set_seed(0)

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
#     loss[loss != loss] = 0
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()

class DCRNNSupervisor:
    def __init__(self):
        with open('data/dcrnn_la.yaml') as f_la, open('data/dcrnn_bay.yaml') as f_bay:
            config_la = yaml.load(f_la, Loader=yaml.FullLoader)
            config_bay = yaml.load(f_bay, Loader=yaml.FullLoader)

        sensor_ids1, sensor_id_to_ind1, adj_mx_la = load_graph_data(config_la['data'].get('graph_pkl_filename'))
        sensor_ids2, sensor_id_to_ind2, adj_mx_bay = load_graph_data(config_bay['data'].get('graph_pkl_filename'))
        
        self._kwargs = config_la
        self._data_kwargs = config_la.get('data')
        self._model_kwargs = config_la.get('model')
        self._data_kwargs2 = config_bay.get('data')
        self._model_kwargs2 = config_bay.get('model')
        self._train_kwargs = config_la.get('train')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # logging.
        self._log_dir = self._get_log_dir(config_la)
        self._writer = SummaryWriter('runs/' + self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        self._data2 = utils.load_dataset(**self._data_kwargs2)
        self.standard_scaler = self._data['scaler']
        self.standard_scaler2 = self._data2['scaler']
        
        self._logger.info('Setting: {}'.format(args.setting))
        self._logger.info("Party A trn samples: {}".format(self._data['train_loader'].size))
        self._logger.info("Party A vld samples: {}".format(self._data['val_loader'].size))
        self._logger.info("Party A tst samples: {}".format(self._data['test_loader'].size))
        self._logger.info("Party B trn samples: {}".format(self._data2['train_loader'].size))
        self._logger.info("Party B vld samples: {}".format(self._data2['val_loader'].size))
        self._logger.info("Party B tst samples: {}".format(self._data2['test_loader'].size))

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.num_nodes2 = int(self._model_kwargs2.get('num_nodes', 1))
        self._logger.info("num_nodes: {}".format(self.num_nodes))
        self._logger.info("num_nodes2: {}".format(self.num_nodes2))

        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # setup model
        dcrnn_model = DCRNNModel(adj_mx_la, self._logger, **self._model_kwargs)
        dcrnn_model2 = DCRNNModel(adj_mx_bay, self._logger, **self._model_kwargs2)

        if torch.cuda.is_available():
            # dcrnn_model = nn.DataParallel(dcrnn_model)
            # dcrnn_model2 = nn.DataParallel(dcrnn_model2)
            self.dcrnn_model = dcrnn_model.cuda()
            self.dcrnn_model2 = dcrnn_model2.cuda()
        else:
            self.dcrnn_model = dcrnn_model
            self.dcrnn_model2 = dcrnn_model2
        self._logger.info("Models created")
        self._logger.info('Local epochs:' + str(args.local_epochs))

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model(self._epoch_num)
            
        # use PySyft for SPDZ
        if args.setting == 'fedavg' and args.spdz:
            import syft as sy
            self._logger.info('Using SPDZ for FedAvg')
            hook = sy.TorchHook(torch)
            self.party_workers = [sy.VirtualWorker(hook, id="party{:d}".format(i)) for i in range(2)]
            self.crypto = sy.VirtualWorker(hook, id="crypto")
            
        # DP
        if args.dp:
            class HiddenPrints:
                def __enter__(self):
                    self._original_stdout = sys.stdout
                    sys.stdout = open(os.devnull, 'w')

                def __exit__(self, exc_type, exc_val, exc_tb):
                    sys.stdout.close()
                    sys.stdout = self._original_stdout

            def find_sigma(eps, batches_per_lot, dataset_size):
                lotSize = batches_per_lot * args.batch_size  # L
                N = dataset_size
                delta = min(10**(-5), 1 / N)
                lotsPerEpoch = N / lotSize
                q = lotSize / N  # Sampling ratio
                T = args.epochs * lotsPerEpoch  # Total number of lots

                def compute_dp_sgd_wrapper(_sigma):
                    with HiddenPrints():
                        return compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=N, batch_size=lotSize, noise_multiplier=_sigma, epochs=args.epochs, delta=delta)[0] - args.epsilon

                sigma = newton(compute_dp_sgd_wrapper, x0=0.5, tol=1e-4)  # adjust x0 to avoid error
                with HiddenPrints():
                    actual_eps = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=N, batch_size=lotSize, noise_multiplier=sigma, epochs=args.epochs, delta=delta)[0]
        #         print('Batches_per_lot={}, q={}, T={}, sigma={}'.format(batches_per_lot, q, T, sigma))
        #         print('actual epslion = {}'.format(actual_eps))
                return sigma

            self._logger.info('Epsilon: ' + str(args.epsilon))
            self._logger.info('Lotsize_scaler: ' + str(args.lotsize_scaler))
            lotsizes = [N**.5 * args.lotsize_scaler for N in [self._data['train_loader'].size, self._data2['train_loader'].size]]
            batches_per_lot_list = list(map(lambda lotsize: max(round(lotsize / args.batch_size), 1), lotsizes))
            batches_per_lot_list = [min(bpl, loader_len) for bpl, loader_len in zip(batches_per_lot_list, [self._data['train_loader'].num_batch, self._data2['train_loader'].num_batch])]
            self._logger.info('Batches per lot: ' + str(batches_per_lot_list))
            sigma_list = [find_sigma(args.epsilon, bpl, N) for bpl, N in zip(batches_per_lot_list, [self._data['train_loader'].size, self._data2['train_loader'].size])]
            self._logger.info('Sigma: ' + str(sigma_list))

            for mod, bpl, sig in zip([self.dcrnn_model, self.dcrnn_model2], batches_per_lot_list, sigma_list):
                mod.batch_per_lot = bpl
                mod.sigma = sig

            self.dcrnn_model.batch_per_lot = batches_per_lot_list[0]
            self.dcrnn_model.sigma = sigma_list[0]
            self.dcrnn_model2.batch_per_lot = batches_per_lot_list[1]
            self.dcrnn_model2.sigma = sigma_list[1]

            self._lastNoiseShape = None
            self._noiseToAdd = None

    def divide_clip_grads(self, model, batch_per_lot=None):
        for key, param in model.named_parameters():
            if batch_per_lot is None:
                param.grad /= model.batch_per_lot
            else:
                param.grad /= batch_per_lot
            nn.utils.clip_grad_norm([param], args.clip)

    def gaussian_noise(self, model, grads):
        if grads.shape != self._lastNoiseShape:
            self._lastNoiseShape = grads.shape
            self._noiseToAdd = torch.zeros(grads.shape).to(device)
        self._noiseToAdd.data.normal_(0.0, std=args.clip*model.sigma)
        return self._noiseToAdd

    def add_noise_to_grads(self, model, batch_per_lot=None):
        for key, param in model.named_parameters():
            if batch_per_lot is None:
                lotsize = model.batch_per_lot * args.batch_size
            else:
                lotsize = batch_per_lot * args.batch_size
            noise = 1/lotsize * self.gaussian_noise(model, param.grad)
            param.grad += noise

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
#         if not os.path.exists('models/'):
#             os.makedirs('models/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.dcrnn_model.state_dict()
        config['epoch'] = epoch
#         torch.save(config, 'models/epo%d.tar' % epoch)
        torch.save(config, self._log_dir + '/epo%d.tar' % epoch)
#         self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self, epo=0):
        self._setup_graph()
        assert os.path.exists('models/epo%d.tar' % epo), 'Weights at epoch %d not found' % epo
        checkpoint = torch.load('models/epo%d.tar' % epo, map_location='cpu')
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.dcrnn_model2.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(epo))

    def _setup_graph(self):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()
            self.dcrnn_model2 = self.dcrnn_model2.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                break
            
            val_iterator = self._data2['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model2(x)
                break
                
    def fedavg(self):
        assert args.setting == 'fedavg'
        self._logger.info('Fedavg now' + (' using SPDZ' if args.spdz else ''))
        model1, model2 = self.dcrnn_model, self.dcrnn_model2
        len1 = self._data['train_loader'].size
        len2 = self._data2['train_loader'].size
        len_total = len1 + len2
        if args.spdz:
            ratio_total = 10000
            ratio1 = round(ratio_total * len1 / len_total)  # integer share used by SPDZ
            ratio2 = round(ratio_total * len2 / len_total)
            new_params = list()
            params = [list(model1.parameters()), list(model2.parameters())]
            for param_i in range(len(params[0])):
                spdz_params = list()
                spdz_params.append(params[0][param_i].copy().cpu().fix_precision().share(*self.party_workers, crypto_provider=self.crypto))
                spdz_params.append(params[1][param_i].copy().cpu().fix_precision().share(*self.party_workers, crypto_provider=self.crypto))
                new_param = (spdz_params[0] * ratio1 + spdz_params[1] * ratio2).get().float_precision() / ratio_total
                new_params.append(new_param)

            with torch.no_grad():
                for model_params in params:
                    for param in model_params:
                        param *= 0

                for param_index in range(len(params[0])):
                    if str(device) == 'cpu':
                        for model_params in params:
                            model_params[param_index].copy_(new_params[param_index])
                    else:
                        for model_params in params:
                            model_params[param_index].copy_(new_params[param_index].cuda())
        else:
            with torch.no_grad():
                for p1, p2 in zip(model1.parameters(), model2.parameters()):
                    data = (p1.data * len1 + p2.data * len2) / len_total
                    p1.copy_(data)
                    p2.copy_(data)

    def assign_weight(self, src, dst):
        with torch.no_grad():
            for p1, p2 in zip(src.parameters(), dst.parameters()):
                p2.copy_(p1.data)

    def train(self):
        return self._train(**self._train_kwargs)

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):

        epochs = args.epochs
        self._setup_graph()
        if args.setting == 'fedavg':
            self.fedavg()

        min_val_loss = float('inf')
        wait = 0

        optimizer1 = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)
        lr_scheduler1 = ReduceLROnPlateau(optimizer1, factor=np.sqrt(0.1), patience=5, verbose=True, threshold=1e-4)
        optimizer2 = torch.optim.Adam(self.dcrnn_model2.parameters(), lr=base_lr, eps=epsilon)
        lr_scheduler2 = ReduceLROnPlateau(optimizer2, factor=np.sqrt(0.1), patience=5, verbose=True, threshold=1e-4)
        schedulers = [lr_scheduler1, lr_scheduler2]
        
        num_batches = self._data['train_loader'].num_batch + self._data2['train_loader'].num_batch
        batches_seen = num_batches * self._epoch_num

        for epoch_num in range(self._epoch_num, epochs):
            epoch = epoch_num
            self._logger.info('Epoch {}'.format(epoch_num))
            losses = []
            
            if args.setting != 'combined':
                for dcrnn_model, data, optimizer, lr_scheduler, compute_loss in \
                    [(self.dcrnn_model, self._data, optimizer1, lr_scheduler1, self._compute_loss), \
                    (self.dcrnn_model2, self._data2, optimizer2, lr_scheduler2, self._compute_loss2)]:
                    # this will fail if model is loaded with a changed batch_size
                    # self._logger.info("num_batches:{}".format(num_batches))

                    if args.setting == '0' and data == self._data2:  # skip a party
                        continue
                    if args.setting == '1' and data == self._data:
                        continue

                    dcrnn_model = dcrnn_model.train()
                    train_iterator = data['train_loader'].get_iterator()

                    for batch_i, (x, y) in enumerate(train_iterator):
                        if args.dp:
                            if batch_i % dcrnn_model.batch_per_lot == 0:
                                optimizer.zero_grad()
                        else:
                            optimizer.zero_grad()
                        x, y = self._prepare_data(x, y)
                        output = dcrnn_model(x, y, batches_seen)
                        if batches_seen == 0:
                            # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                            optimizer = torch.optim.Adam(dcrnn_model.parameters(), lr=base_lr, eps=epsilon)
                        loss = compute_loss(y, output)
                        self._logger.debug(loss.item())
                        losses.append(loss.item())
                        batches_seen += 1
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(dcrnn_model.parameters(), self.max_grad_norm)
                        if args.dp:
                            if batch_i % dcrnn_model.batch_per_lot == dcrnn_model.batch_per_lot - 1:
                                self.divide_clip_grads(dcrnn_model)
                                self.add_noise_to_grads(dcrnn_model)
                                optimizer.step()
                            elif (batch_i == data['train_loader'].size // args.batch_size):  # reach the end of the last incomplete lot
                                self.divide_clip_grads(dcrnn_model, batch_i % dcrnn_model.batch_per_lot + 1)
                                self.add_noise_to_grads(dcrnn_model, batch_i % dcrnn_model.batch_per_lot + 1)
                                optimizer.step()
                        else:
                            optimizer.step()
                if args.setting == 'fedavg' and (epoch % args.local_epochs == args.local_epochs - 1 or epoch == epochs - 1):
                    self.fedavg()
            else:  # combined
                LA_done = False
                BAY_done = False
                train_iterator1 = self._data['train_loader'].get_iterator()
                train_iterator2 = self._data2['train_loader'].get_iterator()
                while not LA_done or not BAY_done:  # train a batch
                    if not LA_done and not BAY_done:
                        party = random.choice([0,1])
                    elif LA_done:
                        party = 1
                    elif BAY_done:
                        party = 0
                    if party == 0:
                        batch = next(train_iterator1, None)
                        compute_loss = self._compute_loss
                        dcrnn_model = self.dcrnn_model.train()
                        optimizer = optimizer1
                        if batch is None:
                            LA_done = True
                            continue
                    else:
                        batch = next(train_iterator2, None)
                        compute_loss = self._compute_loss2
                        dcrnn_model = self.dcrnn_model2.train()
                        optimizer = optimizer2
                        if batch is None:
                            BAY_done = True
                            continue
                    x, y = batch
                    optimizer.zero_grad()
                    x, y = self._prepare_data(x, y)
                    output = dcrnn_model(x, y, batches_seen)
                    if batches_seen == 0:
                        # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                        optimizer = torch.optim.Adam(dcrnn_model.parameters(), lr=base_lr, eps=epsilon)
                    loss = compute_loss(y, output)
                    self._logger.debug(loss.item())
                    losses.append(loss.item())
                    batches_seen += 1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(dcrnn_model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    if party == 0:
                        self.assign_weight(self.dcrnn_model, self.dcrnn_model2)
                    else:
                        self.assign_weight(self.dcrnn_model2, self.dcrnn_model)
            

            # VLD
            val_loss1, dict1, val_loss2, dict2 = self.evaluate(dataset='val', batches_seen=batches_seen)
            
            val_loss = (val_loss1 * self._data['val_loader'].size + val_loss2 * self._data2['val_loader'].size) / (self._data['val_loader'].size + self._data2['val_loader'].size)

            message = 'EVAL trn_loss: {:.4f}, val_loss_la: {:.4f}, val_loss_bay: {:.4f}, val_loss: {:.4f}' \
                    .format(np.mean(losses), val_loss1, val_loss2, val_loss)
            self._logger.info(message)
            
            for s in schedulers:
                s.step(val_loss)
            
            mae1 = self.mae_at_T(dict1['prediction'], dict1['truth'], T=3)
            mae2 = self.mae_at_T(dict2['prediction'], dict2['truth'], T=3)
            message = 'EVAL val_mae3_la: {:.2f}, val_mae3_bay: {:.2f}'.format(mae1, mae2)
            self._logger.info(message)
            mae1 = self.mae_at_T(dict1['prediction'], dict1['truth'], T=6)
            mae2 = self.mae_at_T(dict2['prediction'], dict2['truth'], T=6)
            message = 'EVAL val_mae6_la: {:.2f}, val_mae6_bay: {:.2f}'.format(mae1, mae2)
            self._logger.info(message)
            mae1 = self.mae_at_T(dict1['prediction'], dict1['truth'], T=12)
            mae2 = self.mae_at_T(dict2['prediction'], dict2['truth'], T=12)
            message = 'EVAL val_mae12_la: {:.2f}, val_mae12_bay: {:.2f}'.format(mae1, mae2)
            self._logger.info(message)

            self._writer.add_scalar('training loss', np.mean(losses), batches_seen)

            # TST
            tst_loss1, dict1, tst_loss2, dict2 = self.evaluate(dataset='test', batches_seen=batches_seen)
            mae1 = self.mae_at_T(dict1['prediction'], dict1['truth'])
            mae2 = self.mae_at_T(dict2['prediction'], dict2['truth'])
            message = 'TEST tst_loss_la: {:.4f}, tst_loss_bay: {:.4f}'.format(tst_loss1, tst_loss2)
            self._logger.info(message)
            
            mae1 = self.mae_at_T(dict1['prediction'], dict1['truth'], T=3)
            mae2 = self.mae_at_T(dict2['prediction'], dict2['truth'], T=3)
            message = 'TEST val_mae3_la: {:.2f}, val_mae3_bay: {:.2f}'.format(mae1, mae2)
            self._logger.info(message)
            mae1 = self.mae_at_T(dict1['prediction'], dict1['truth'], T=6)
            mae2 = self.mae_at_T(dict2['prediction'], dict2['truth'], T=6)
            message = 'TEST val_mae6_la: {:.2f}, val_mae6_bay: {:.2f}'.format(mae1, mae2)
            self._logger.info(message)
            mae1 = self.mae_at_T(dict1['prediction'], dict1['truth'], T=12)
            mae2 = self.mae_at_T(dict2['prediction'], dict2['truth'], T=12)
            message = 'TEST val_mae12_la: {:.2f}, val_mae12_bay: {:.2f}'.format(mae1, mae2)
            self._logger.info(message)

            if min_val_loss - val_loss >= 1e-4:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
            else:
                wait += 1
                if wait == patience:
                    self._logger.info('Early stopping at epoch: %d' % epoch_num)
                    break

    def evaluate(self, dataset='val', batches_seen=0):  # dataset = val|test
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        mean_losses = []
        dicts = []
        for dcrnn_model, data, scaler, compute_loss in [(self.dcrnn_model, self._data, self.standard_scaler, self._compute_loss), (self.dcrnn_model2, self._data2, self.standard_scaler2, self._compute_loss2)]:
            with torch.no_grad():
                dcrnn_model = dcrnn_model.eval()

                val_iterator = data['{}_loader'.format(dataset)].get_iterator()
                losses = []

                y_truths = []
                y_preds = []

                for _, (x, y) in enumerate(val_iterator):
                    x, y = self._prepare_data(x, y)

                    output = dcrnn_model(x)
                    loss = compute_loss(y, output)
                    losses.append(loss.item())

                    y_truths.append(y.cpu())
                    y_preds.append(output.cpu())

                mean_loss = np.mean(losses)

                self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

                y_preds = np.concatenate(y_preds, axis=1)
                y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

                y_truths_scaled = []
                y_preds_scaled = []
                for t in range(y_preds.shape[0]):
                    y_truth = scaler.inverse_transform(y_truths[t])
                    y_pred = scaler.inverse_transform(y_preds[t])
                    y_truths_scaled.append(y_truth)
                    y_preds_scaled.append(y_pred)

                mean_losses.append(mean_loss)
                dicts.append({'prediction': y_preds_scaled, 'truth': y_truths_scaled})

        return mean_losses[0], dicts[0], mean_losses[1], dicts[1]

    def _prepare_data(self, x, y):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        # x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        # y = y[..., :self.output_dim].view(self.horizon, batch_size, self.num_nodes * self.output_dim)
        x = x.view(self.seq_len, batch_size, -1)
        y = y[..., :self.output_dim].view(self.horizon, batch_size, -1)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)

    def _compute_loss2(self, y_true, y_predicted):
        y_true = self.standard_scaler2.inverse_transform(y_true)
        y_predicted = self.standard_scaler2.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)

    def mae_at_T(self, preds, labels, T=12):  # T = horizon = number of 5-min intervals
        preds = preds[T-1]
        labels = labels[T-1]
        null_val = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            if np.isnan(null_val):
                mask = ~np.isnan(labels)
            else:
                mask = np.not_equal(labels, null_val)
            mask = mask.astype('float32')
            mask /= np.mean(mask)
            mae = np.abs(np.subtract(preds, labels)).astype('float32')
            mae = np.nan_to_num(mae * mask)
            return np.mean(mae)

if __name__ == '__main__':
    supervisor_combined = DCRNNSupervisor()
    supervisor_combined.train()

# TEST
# loss1, la, loss2, bay = supervisor_combined.evaluate('test')
# print("Loss_LA : {}, Loss_BAY : {}".format(loss1, loss2))
# np.savez_compressed('data/predictions_la.npz', **la)
# np.savez_compressed('data/predictions_bay.npz', **bay)
# la = np.load('data/predictions_la.npz')
# bay = np.load('data/predictions_bay.npz')

# def masked_mae_np(preds, labels, null_val=0):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         if np.isnan(null_val):
#             mask = ~np.isnan(labels)
#         else:
#             mask = np.not_equal(labels, null_val)
#         mask = mask.astype('float32')
#         mask /= np.mean(mask)
#         mae = np.abs(np.subtract(preds, labels)).astype('float32')
#         mae = np.nan_to_num(mae * mask)
#         return np.mean(mae)

# pred = la['prediction']
# truth = la['truth']
# print(masked_mae_np(pred, truth))
# print(masked_mae_np(pred[2], truth[2]))
# print(masked_mae_np(pred[5], truth[5]))
# print(masked_mae_np(pred[11], truth[11]))
# print()

# pred = bay['prediction']
# truth = bay['truth']
# print(masked_mae_np(pred, truth))
# print(masked_mae_np(pred[2], truth[2]))
# print(masked_mae_np(pred[5], truth[5]))
# print(masked_mae_np(pred[11], truth[11]))
