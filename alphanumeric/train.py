import numpy as np
from scipy.optimize import newton
import argparse
import time
import copy
import math
import sys
import os
import random

parser = argparse.ArgumentParser(description='Alphanumeric Character Recognition')
parser.add_argument('--task-dir', default='.', help="ML task directory")
parser.add_argument('--setting', default='fedavg', help="Training setting (0|1|...|combined|fedavg), where the number represents the party")
parser.add_argument('--dp', action='store_true', default=False, help='Enable DP')
parser.add_argument('--spdz', action='store_true', default=False, help='Enable SPDZ')
parser.add_argument('--pysyft-remote-training', action='store_true', default=False, help='Enable PySyft remote training (not supported for now)')

parser.add_argument('-e', '--epsilon', default=1.0, type=float, help="Privacy Budget for each party")  # experiment variable
parser.add_argument('--lotsize-scaler', default=1.0, type=float, help="Scale the lot size sqrt(N) by a multiplier")  # experiment variable
parser.add_argument('-c', '--clip', default=1.0, type=float, help="L2 bound for the gradient clip")  # experiment variable
parser.add_argument('--local-epochs', default=1, type=int, help="FedAvg per how many epochs")  # experiment variable

parser.add_argument('-E', '--epochs', default=50, type=int, help="Number of epochs")
parser.add_argument('-b', '--batch-size', default=256, type=int, help="How many records per batch")
parser.add_argument('--val-batch-size', default=64, type=int, help="Validation and testing set batch size")
parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
parser.add_argument('--patience', default=5, type=int, help="Patience for early stopping")
parser.add_argument('--patience-reducelr', default=2, type=int, help="Patience for ReduceLROnPlateau")
parser.add_argument('--min_delta', default=1e-4, type=float, help="Min delta for early stopping and ReduceLROnPlateau")
parser.add_argument('--save-model', action='store_true', default=False, help="Save models")
parser.add_argument('--save-freq', default=0, type=int, help="Save model per how many epochs")

parser.add_argument('--gpu', action='store_true', default=True, help="Use gpu")
parser.add_argument('--which-gpu', default="0", help="Use which gpu")
parser.add_argument('--seed', default=123, type=int, help="Random seed")
parser.add_argument('--load-model', default=None, help="Load trained model, e.g. chinese/models/epo19.pt")
parser.add_argument('--starting-epoch', default=0, type=int, help="Start from the beginning of which epoch, e.g. 20")

parser.add_argument('--debug', action='store_true', default=False, help="Use small subsets to debug")

args = parser.parse_args()

assert args.setting in ['combined', 'fedavg'] or args.setting.isdigit(), 'Setting not supported'
if args.spdz: assert args.setting == 'fedavg'
assert args.pysyft_remote_training == False, 'PySyft remote training not supported for now'

os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpu

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import h5py as h5
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import RandomSampler, DataLoader, Dataset, ConcatDataset
from torch.autograd import Variable
import syft as sy

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(args.seed)


from model import get_model, get_loss_func, get_metric_func
from dataset import get_datasets

loss_func = get_loss_func(args)
metric_func = get_metric_func(args)

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.gpu else {}

print('Device:', device)
print('Setting:', args.setting)
if args.spdz: print('Using SPDZ for FedAvg')
if args.dp: print('Using DP')
print('Local epochs:', args.local_epochs)
print('Epochs', args.epochs)

trn_party_datasets, val_party_datasets, tst_party_datasets = get_datasets(args)
assert len(trn_party_datasets) == len(val_party_datasets) and len(val_party_datasets) == len(tst_party_datasets)
num_parties = len(trn_party_datasets)
assert num_parties > 1

trn_party_loaders = [DataLoader(trnset, batch_size=args.batch_size, shuffle=True, **kwargs) for trnset in trn_party_datasets]

if args.setting == 'fedavg':
    trn_loaders = trn_party_loaders
elif args.setting == 'combined':
    trn_combined_dataset = ConcatDataset(trn_party_datasets)
    trn_combined_loader = DataLoader(trn_combined_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    trn_loaders = [trn_combined_loader]
else:
    party = int(args.setting)
    trn_loaders = [trn_party_loaders[party]]

val_loaders = [DataLoader(valset, batch_size=args.val_batch_size, shuffle=False, **kwargs) for valset in val_party_datasets]
tst_loaders = [DataLoader(tstset, batch_size=args.val_batch_size, shuffle=False, **kwargs) for tstset in tst_party_datasets]

print('Trn sizes:', [len(loader.dataset) for loader in trn_loaders])
print('Trn batchs:', [len(loader) for loader in trn_loaders])
print('Val sizes:', [len(loader.dataset) for loader in val_loaders])
print('Val batchs:', [len(loader) for loader in val_loaders])
print('Tst sizes:', [len(loader.dataset) for loader in tst_loaders])
print('Tst batchs:', [len(loader) for loader in tst_loaders])

trn_lengths = [len(loader.dataset) for loader in trn_loaders]  # party's trn dataset lengths
val_lengths = [len(loader.dataset) for loader in val_loaders]  # party's val dataset lengths
tst_lengths = [len(loader.dataset) for loader in tst_loaders]  # party's tst dataset lengths
num_trn_parties = len(trn_lengths)

if args.setting != 'fedavg': assert num_trn_parties == 1
else: assert num_trn_parties > 1

def avg(numbers, weights):  # weighted average to compute weighted metric or loss
    assert len(numbers) == num_parties and len(weights) == num_parties, "length must be the number of parties"
    return sum([n * l for n, l in zip(numbers, weights)]) / sum(weights)

precision = 10000
ratios = [round(precision * length / sum(trn_lengths)) for length in trn_lengths]  # integer share used by SPDZ
if args.spdz:
    print('SPDZ fedavg ratios:', ratios)

batches_per_lot_list = [None] * num_trn_parties
sigma_list = [None] * num_trn_parties
if args.dp:
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    def find_sigma(eps, batches_per_lot, dataset_size):
        lotSize = batches_per_lot * args.batch_size # L
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
        print('N={}, lotsize={}, delta={}, lotsPerEpoch={}, q={}, T={}, sigma={}'.format(N, lotSize, delta, lotsPerEpoch, q, T, sigma))
        print('Actual epslion = {}'.format(actual_eps))
        return sigma

    print('Epsilon:', args.epsilon)
    lotsizes = [len(loader.dataset)**.5 * args.lotsize_scaler for loader in trn_loaders]
    batches_per_lot_list = list(map(lambda lotsize: max(round(lotsize / args.batch_size), 1), lotsizes))
    batches_per_lot_list = [min(bpl, len(loader)) for bpl, loader in zip(batches_per_lot_list, trn_loaders)]
    print('Batches per lot:', batches_per_lot_list)
    sigma_list = [find_sigma(args.epsilon, bpl, len(loader.dataset)) for bpl, loader in zip(batches_per_lot_list, trn_loaders)]
    print('Sigma:', sigma_list)

_lastNoiseShape = None
_noiseToAdd = None

def divide_clip_grads(model, batch_per_lot=None):
    assert args.dp == True
    for key, param in model.named_parameters():
        if batch_per_lot is None:
            param.grad /= model.batch_per_lot
        else:
            param.grad /= batch_per_lot
        nn.utils.clip_grad_norm([param], args.clip)

def gaussian_noise(model, grads):
    global _lastNoiseShape
    global _noiseToAdd
    if grads.shape != _lastNoiseShape:
        _lastNoiseShape = grads.shape
        _noiseToAdd = torch.zeros(grads.shape).to(device)
    _noiseToAdd.data.normal_(0.0, std=args.clip*model.sigma)
    return _noiseToAdd

def add_noise_to_grads(model, batch_per_lot=None):
    assert args.dp == True
    for key, param in model.named_parameters():
        if batch_per_lot is None:
            lotsize = model.batch_per_lot * args.batch_size
        else:
            lotsize = batch_per_lot * args.batch_size
        noise = 1/lotsize * gaussian_noise(model, param.grad)
        param.grad += noise

model = get_model(args)

if args.load_model is not None:
    model.load_state_dict(torch.load(args.load_model))
    print('Loaded model:', args.load_model)

models = [model]
models += [copy.deepcopy(model) for _ in range(num_trn_parties - 1)]

assert len(models) == num_trn_parties

if args.dp:
    assert len(batches_per_lot_list) == num_trn_parties
    assert len(sigma_list) == num_trn_parties
    for mod, bpl, sig in zip(models, batches_per_lot_list, sigma_list):
        mod.batch_per_lot = bpl
        mod.sigma = sig

optims = []
schedulers = []
for i in range(len(models)):
    models[i] = models[i].to(device)
    optimizer = optim.Adam(models[i].parameters(), lr=args.lr)
    optims.append(optimizer)
    scheduler = ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=args.patience_reducelr, verbose=True, threshold=args.min_delta)
    schedulers.append(scheduler)
    params = [list(mod.parameters()) for mod in models]

if args.setting == 'fedavg' and args.spdz:
    # use PySyft for SPDZ
    hook = sy.TorchHook(torch)
    party_workers = [sy.VirtualWorker(hook, id="party{:d}".format(i)) for i in range(num_trn_parties)]
    crypto = sy.VirtualWorker(hook, id="crypto")

avg_flag = True
def fedavg():
    assert num_trn_parties > 1 and args.setting == 'fedavg'
    print('Fedavg now')
    if args.spdz:
        new_params = list()
        for param_i in range(len(params[0])):
            spdz_params = list()
            for party_i in range(num_trn_parties):
                spdz_params.append(params[party_i][param_i].copy().cpu().fix_precision(precision_fractional=4).share(*party_workers, crypto_provider=crypto))
            new_param = sum([p * r for p, r in zip(spdz_params, ratios)]).get().float_precision() / sum(ratios)
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
                        model_params[param_index].set_(new_params[param_index].cuda())
    else:
        with torch.no_grad():
            for ps in zip(*params):
                # p_avg = sum([p.data * r for p, r in zip(ps, ratios)]) / sum(ratios)  # coarse
                p_avg = avg([p.data for p in ps], trn_lengths)
                for p in ps:
                    p.copy_(p_avg)

    global avg_flag
    avg_flag = True

def train(epoch):
    for mod in models:
        mod.train()
    
    losses = [0] * num_trn_parties
    metrics = [0] * num_trn_parties
    
    def trn_batch(data, target, model, optimizer, party_i, batch_i, batch_per_lot):
        data, target = data.to(device), target.to(device)
        
        if args.dp:
            if batch_i % batch_per_lot == 0:
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()
        
        output = model(data)
        loss = loss_func(output, target)
        losses[party_i] += loss_func(output, target, reduction='sum').item()
        metrics[party_i] += metric_func(output, target)
        loss.backward()
        
        if args.dp:
            if batch_i % batch_per_lot == batch_per_lot - 1:
                divide_clip_grads(model)
                add_noise_to_grads(model)
                optimizer.step()
            elif (batch_i == len(trn_loaders[party_i]) - 1):  # reach the end of the last incomplete lot
                divide_clip_grads(model, batch_i % batch_per_lot + 1)
                add_noise_to_grads(model, batch_i % batch_per_lot + 1)
                optimizer.step()
        else:
            optimizer.step()
    
    for party_i, (loader, mod, optim, bpl) in enumerate(zip(trn_loaders, models, optims, batches_per_lot_list)):
        for batch_i, (data, target) in enumerate(loader):
            trn_batch(data, target, mod, optim, party_i, batch_i, bpl)
    
    loss_print = 'Trn loss: '
    metric_print = 'Trn metric: '
    for i in range(num_trn_parties):
        losses[i] /= len(trn_loaders[i].dataset)
        metrics[i] /= len(trn_loaders[i].dataset)
        loss_print += '{:4f} '.format(losses[i])
        metric_print += '{:4f} '.format(metrics[i])
    print(loss_print)
    print(metric_print)

    global avg_flag
    avg_flag = False

    if args.setting == 'fedavg' and (epoch % args.local_epochs == args.local_epochs - 1 or epoch == args.epochs - 1):
        fedavg()

min_loss = float('inf')
wait = 0

def val(loaders):
    if avg_flag:
        assert len(params[0]) == len(params[1])
        for param_index in range(len(params[0])):
            assert torch.equal(params[0][param_index], params[1][param_index])

    for mod in models:
        mod.eval()
    losses = [0] * num_parties
    metrics = [0] * num_parties
    
    def val_batch(data, target, model, party_i):
        data, target = data.to(device), target.to(device)
        output = model(data)
        losses[party_i] += loss_func(output, target, reduction='sum').item()
        metrics[party_i] += metric_func(output, target)
    
    for party_i, loader in enumerate(loaders):
        for data, target in loader:
            val_batch(data, target, models[party_i % num_trn_parties], party_i)

    if loaders == val_loaders:
        split = 'Val'
        lengths = val_lengths
    elif loaders == tst_loaders:
        split = 'Tst'
        lengths = tst_lengths
    else:
        split = 'Trn'
        lengths = trn_lengths
    loss_print = '{} loss: '.format(split)
    metric_print = '{} metric: '.format(split)
    for i in range(num_parties):
        losses[i] /= len(loaders[i].dataset)
        metrics[i] /= len(loaders[i].dataset)
        loss_print += '{:4f} '.format(losses[i])
        metric_print += '{:4f} '.format(metrics[i])
    print(loss_print)
    print(metric_print)
    
    loss_avg = avg(losses, lengths)
    metric_avg = avg(metrics, lengths)
    print('{} loss_avg {:.4f}, metric_avg {:.4f}'.format(split, loss_avg, metric_avg))
    
    if loaders == val_loaders:
        global min_loss
        global wait
        if min_loss - loss_avg > args.min_delta:
            min_loss = loss_avg
            wait = 0
        else:
            wait += 1

        for s in schedulers:
            s.step(loss_avg)

# save model to
if args.save_model:
    model_dir = '{}/models/setting-{}-localepochs-{}'.format(args.task_dir, args.setting, args.local_epochs)
    if args.dp:
        model_dir += '-eps-{}-lotsize_scaler-{}'.format(args.epsilon, args.lotsize_scaler)
    if args.spdz:
        model_dir += '-spdz'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
# Train
cnt = 0
for epoch in range(args.starting_epoch, args.epochs):
    print('Epoch', epoch)
    t1 = int(time.time())
    train(epoch)
    t2 = int(time.time())
    val(val_loaders)
    val(tst_loaders)
    if args.save_model and args.save_freq > 0 and epoch % args.save_freq == args.save_freq - 1:
        torch.save(models[0].state_dict(), model_dir + "/epoch-{}.pt".format(epoch))
        print('Saved model to:', model_dir + "/epoch-{}.pt".format(epoch))
    t3 = int(time.time())
    print('Epoch trn time {:d}s, val time {:d}s'.format(t2-t1, t3-t2))
    cnt += 1
    if wait == args.patience:
        print('Wait =', wait)
        print('Early stop')
        if args.setting == 'fedavg':
            fedavg()
            val(val_loaders)
            val(tst_loaders)
        break

if args.save_model:
    torch.save(models[0].state_dict(), model_dir + "/epoch-{}.pt".format(cnt))



