import numpy as np
from scipy.optimize import newton
import argparse
import time
import copy
import math
import sys
import os

parser = argparse.ArgumentParser(description='Handwritten Chinese Character Recognition')
parser.add_argument('--setting', default='fedavg', help="Training setting (casia|hit|combined|fedavg)")
parser.add_argument('--dp', action='store_true', default=False, help='Enable DP')
parser.add_argument('--spdz', action='store_true', default=False, help='Enable SPDZ')
parser.add_argument('--pysyft-remote-training', action='store_true', default=False, help='Enable PySyft remote training (buggy for now)')
parser.add_argument('-e', '--epsilon', default=1.0, type=float, help="Privacy Budget for each party")  # experiment variable
parser.add_argument('--lotsize-scaler', default=1.0, type=float, help="Scale the lot size sqrt(N) by a multiplier")  # experiment variable
parser.add_argument('-c', '--clip', default=1.0, type=float, help="L2 bound for the gradient clip")  # experiment variable
parser.add_argument('-E', '--epochs', default=10, type=int, help="Number of epochs")
parser.add_argument('--freq', default=1, type=int, help="FedAvg per how many epochs")
parser.add_argument('-b', '--batch-size', default=128, type=int, help="How many records per batch")
parser.add_argument('--val-batch-size', default=256, type=int, help="Validation and testing set batch size")
parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
parser.add_argument('--gpu', action='store_true', default=True, help="Use gpu")
parser.add_argument('--which-gpu', default="0", help="Use which gpu")
parser.add_argument('--seed', default=0, type=int, help="Random seed")
parser.add_argument('--starting-epoch', default=0, type=int, help="Start from which epoch")

args = parser.parse_args()  # parse shell args
assert args.setting in ['casia', 'hit', 'combined', 'fedavg'], 'Setting should be (casia|hit|combined|fedavg)'
if args.spdz:
    assert args.setting == 'fedavg'
assert args.pysyft_remote_training == False, 'Not supported for now'

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
from torch.utils.data import RandomSampler, DataLoader, Dataset
from torch.autograd import Variable
import syft as sy

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
print(device)
print('Setting:', args.setting)
if args.spdz:
    print('Using SPDZ for FedAvg')
print('Freq:', args.freq)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.gpu else {}

# load data
class HCC(Dataset):  # Handwritten Chinese character dataset
    # group: trn/vld
    def __init__(self, archive, group, transform=None):
        self.trn = True if group == 'trn' else False
        self.archive = h5.File(archive, 'r')
        self.x = self.archive[group + '/x'][:1024]  # debug
        self.y = self.archive[group + '/y'][:1024]
        self.transform = transform
    def __getitem__(self, index):
        datum = self.x[index]
        if self.transform is not None:
            datum = self.transform(datum)
        label = self.y[index][0].astype('int64')
        return datum, label
    def __len__(self):
        if self.trn and args.dp:
            return len(self.y) - len(self.y) % args.batch_size
        else:
            return len(self.y)
    def close(self):
        self.archive.close()

def to_tensor(img):
    img = torch.from_numpy(img)
    return img.float().div(255)
tfm = transforms.Lambda(to_tensor)

trainset_casia = HCC('data/HWDB1.1fullset.hdf5', 'trn', transform=transforms.Compose([tfm]))
trainset_hit = HCC('data/HIT_OR3Cfullset.hdf5', 'trn', transform=transforms.Compose([tfm]))
trainset_combined = HCC('data/HIT_HWDB1.1_fullset.hdf5', 'trn', transform=transforms.Compose([tfm]))
if args.dp:
    sampler_casia = RandomSampler(trainset_casia, replacement=True)
    sampler_hit = RandomSampler(trainset_hit, replacement=True)
    sampler_combined = RandomSampler(trainset_combined, replacement=True)
    train_loader_casia = DataLoader(trainset_casia, batch_size=args.batch_size, shuffle=False, sampler=sampler_casia, **kwargs)
    train_loader_hit = DataLoader(trainset_hit, batch_size=args.batch_size, shuffle=False, sampler=sampler_hit, **kwargs)
    train_loader_combined = DataLoader(trainset_combined, batch_size=args.batch_size, shuffle=False, sampler=sampler_hit, **kwargs)
else:
    train_loader_casia = DataLoader(trainset_casia, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader_hit = DataLoader(trainset_hit, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader_combined = DataLoader(trainset_combined, batch_size=args.batch_size, shuffle=True, **kwargs)

valset_casia = HCC('data/HWDB1.1fullset.hdf5', 'vld', transform=transforms.Compose([tfm]))
valset_hit = HCC('data/HIT_OR3Cfullset.hdf5', 'vld', transform=transforms.Compose([tfm]))
valset_combined = HCC('data/HIT_HWDB1.1_fullset.hdf5', 'vld', transform=transforms.Compose([tfm]))
val_loader_casia = DataLoader(valset_casia, batch_size=args.val_batch_size, shuffle=False, **kwargs)
val_loader_hit = DataLoader(valset_hit, batch_size=args.val_batch_size, shuffle=False, **kwargs)
val_loader_combined = DataLoader(valset_combined, batch_size=args.val_batch_size, shuffle=False, **kwargs)

tstset_casia = HCC('data/HWDB1.1fullset.hdf5', 'tst', transform=transforms.Compose([tfm]))
tstset_hit = HCC('data/HIT_OR3Cfullset.hdf5', 'tst', transform=transforms.Compose([tfm]))
tstset_combined = HCC('data/HIT_HWDB1.1_fullset.hdf5', 'tst', transform=transforms.Compose([tfm]))
tst_loader_casia = DataLoader(tstset_casia, batch_size=args.val_batch_size, shuffle=False, **kwargs)
tst_loader_hit = DataLoader(tstset_hit, batch_size=args.val_batch_size, shuffle=False, **kwargs)
tst_loader_combined = DataLoader(tstset_combined, batch_size=args.val_batch_size, shuffle=False, **kwargs)

print('CASIA train:', len(trainset_casia), len(train_loader_casia))
print('HIT train:', len(trainset_hit), len(train_loader_hit))
print('COMBINED train:', len(trainset_combined), len(train_loader_combined))
print('CASIA val:', len(valset_casia), len(val_loader_casia))
print('HIT val:', len(valset_hit), len(val_loader_hit))
print('COMBINED val:', len(valset_combined), len(val_loader_combined))
print('CASIA tst:', len(tstset_casia), len(tst_loader_casia))
print('HIT tst:', len(tstset_hit), len(tst_loader_hit))
print('COMBINED tst:', len(tstset_combined), len(tst_loader_combined))

ratio_casia = 25  # integer ratio used by SPDZ
ratio_hit = 10

batches_per_lot_casia, sigma_casia, batches_per_lot_hit, sigma_hit, batches_per_lot_combined, sigma_combined = None, None, None, None, None, None
if args.dp:
    lotsize_casia = len(trainset_casia)**.5 * args.lotsize_scaler
    lotsize_hit = len(trainset_hit)**.5 * args.lotsize_scaler
    lotsize_combined = len(trainset_combined)**.5 * args.lotsize_scaler

    batches_per_lot_casia = max(round(lotsize_casia / args.batch_size), 1)
    batches_per_lot_casia = min(batches_per_lot_casia, len(train_loader_casia))
    batches_per_lot_hit = max(round(lotsize_hit / args.batch_size), 1)
    batches_per_lot_hit = min(batches_per_lot_hit, len(train_loader_hit))
    batches_per_lot_combined = max(round(lotsize_combined / args.batch_size), 1)
    batches_per_lot_combined = min(batches_per_lot_combined, len(train_loader_combined))

    if args.setting == 'casia':
        print('batches per lot:', batches_per_lot_casia)
    elif args.setting == 'hit':
        print('batches per lot:', batches_per_lot_hit)
    elif args.setting == 'combined':
        print('batches per lot:', batches_per_lot_combined)
    elif args.setting == 'fedavg':
        print('batches per lot:', batches_per_lot_casia, batches_per_lot_hit)

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
#         print('Batches_per_lot={}, q={}, T={}, sigma={}'.format(batches_per_lot, q, T, sigma))
#         print('actual epslion = {}'.format(actual_eps))
        return sigma
    
    print('Epsilon:', args.epsilon)
    if args.setting in ['fedavg', 'casia']:
        sigma_casia = find_sigma(args.epsilon, batches_per_lot_casia, len(trainset_casia))
        print('Sigma_casia:', sigma_casia)
    if args.setting in ['fedavg', 'hit']:
        sigma_hit = find_sigma(args.epsilon, batches_per_lot_hit, len(trainset_hit))
        print('Sigma_hit:', sigma_hit)
    if args.setting == 'combined':
        sigma_combined = find_sigma(args.epsilon, batches_per_lot_combined, len(trainset_combined))
        print('Sigma_combined', sigma_combined)

# Model
class VGG(nn.Module):
    def __init__(self, features, num_classes, batch_per_lot=None, sigma=None):
        super(VGG, self).__init__()
        self.features = features
        
        self.batch_per_lot = batch_per_lot
        self.sigma = sigma
        self._lastNoiseShape = None
        self._noiseToAdd = None
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024, momentum=0.66),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256, momentum=0.66),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def divide_clip_grads(self, batch_per_lot=None):
        assert args.dp == True
        for key, param in self.named_parameters():
            if batch_per_lot is None:
                param.grad /= self.batch_per_lot
            else:
                param.grad /= batch_per_lot
            nn.utils.clip_grad_norm([param], args.clip)

    def gaussian_noise(self, grads):
        # slow
#         shape = grads.shape
#         noise = Variable(torch.zeros(shape))
#         noise = noise.to(device)
#         noise.data.normal_(0.0, std=args.clip*self.sigma)
#         return noise
    
        if grads.shape != self._lastNoiseShape:
            self._lastNoiseShape = grads.shape
            self._noiseToAdd = torch.zeros(grads.shape).to(device)
        self._noiseToAdd.data.normal_(0.0, std=args.clip*self.sigma)
        return self._noiseToAdd

    def add_noise_to_grads(self, batch_per_lot=None):
        assert args.dp == True
        for key, param in self.named_parameters():
            if batch_per_lot is None:
                lotsize = self.batch_per_lot * args.batch_size
            else:
                lotsize = batch_per_lot * args.batch_size
            noise = 1/lotsize * self.gaussian_noise(param.grad)
            param.grad += noise

def conv_unit(input, output, mp=False):
    if mp:
        return [nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(), 
               nn.BatchNorm2d(output, momentum=0.66), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
    else:
        return [nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(), 
               nn.BatchNorm2d(output, momentum=0.66)]

def make_layers():
    layers = []
    layers += [nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(), 
               nn.BatchNorm2d(64, momentum=0.66)]

    layers += conv_unit(64, 128)
    layers += conv_unit(128, 128, mp=True)

    layers += conv_unit(128, 256)
    layers += conv_unit(256, 256, mp=True)

    layers += conv_unit(256, 384)
    layers += conv_unit(384, 384)
    layers += conv_unit(384, 384, mp=True)

    layers += conv_unit(384, 512)
    layers += conv_unit(512, 512)
    layers += conv_unit(512, 512, mp=True)

    layers += [nn.Flatten()]

    return nn.Sequential(*layers)

# use PySyft for SPDZ
hook = sy.TorchHook(torch)
casia = sy.VirtualWorker(hook, id="casia")
hit = sy.VirtualWorker(hook, id="hit")
crypto = sy.VirtualWorker(hook, id="crypto")

# Uncomment SWITCH ON to switch on PySyft remote training. Currently buggy.

# SWITCH ON
# compute_nodes = [casia, hit]
# remote_loader_casia = []
# remote_loader_hit = []

# for batch_idx, (data,target) in enumerate(train_loader_casia):
#     data = data.send(casia)
#     target = target.send(casia)
#     remote_loader_casia.append((data, target))

# for batch_idx, (data,target) in enumerate(train_loader_hit):
#     data = data.send(hit)
#     target = target.send(hit)
#     remote_loader_hit.append((data, target))

model_casia = VGG(make_layers(), 3755, batches_per_lot_casia, sigma_casia)

if args.starting_epoch > 0:
    model_casia.load_state_dict(torch.load('models/fl-hit-casia-{:d}.pt'.format(args.starting_epoch)))
    print('Loaded model')

model_hit = copy.deepcopy(model_casia)
model_combined = copy.deepcopy(model_casia)

model_hit.batch_per_lot = batches_per_lot_hit
model_hit.sigma = sigma_hit
model_combined.batch_per_lot = batches_per_lot_combined
model_combined.sigma = sigma_combined

model_casia = model_casia.to(device)
model_hit = model_hit.to(device)
model_combined = model_combined.to(device)

optim_casia = optim.Adam(model_casia.parameters(), lr=args.lr)
optim_hit = optim.Adam(model_hit.parameters(), lr=args.lr)
optim_combined = optim.Adam(model_combined.parameters(), lr=args.lr)

models = [model_casia, model_hit, model_combined]
params = [list(model_casia.parameters()), list(model_hit.parameters()), list(model_combined.parameters())]
optims = [optim_casia, optim_hit, optim_combined]

# models[0].send(compute_nodes[0])  # SWITCH ON
# models[1].send(compute_nodes[1])

def train(epoch):
    
    if args.freq == 1:
        assert len(params[0]) == len(params[1])
        for param_index in range(len(params[0])):
            assert torch.equal(params[0][param_index], params[1][param_index])

    models[0].train()
    models[1].train()
    models[2].train()
    
    losses = [0, 0, 0]
    corrects = [0, 0, 0]
    
    def update(data, target, model, optimizer, party, batch_i, batch_per_lot):
        assert party in ['casia', 'hit', 'combined']
        
        data, target = data.to(device), target.to(device)
        
        if args.dp:
            if batch_i % batch_per_lot == 0:
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()
        
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)    # get the index of the max log-probability
        if party == 'casia':
            corrects[0] += pred.eq(target.view_as(pred)).sum().item()  # debug SWITCH OFF
            losses[0] += F.nll_loss(output, target, reduction='sum').item()
#             corrects[0] += pred.eq(target.view_as(pred)).sum().get().item()  # SWITCH ON
#             losses[0] += F.nll_loss(output, target, reduction='sum').get().item()
        elif party == 'hit':
            corrects[1] += pred.eq(target.view_as(pred)).sum().item()  # debug SWITCH OFF
            losses[1] += F.nll_loss(output, target, reduction='sum').item()
#             corrects[1] += pred.eq(target.view_as(pred)).sum().get().item()  # SWITCH ON
#             losses[1] += F.nll_loss(output, target, reduction='sum').get().item()
        elif party == 'combined':
            corrects[2] += pred.eq(target.view_as(pred)).sum().item()
            losses[2] += F.nll_loss(output, target, reduction='sum').item()
        loss.backward()
        
        if args.dp:
            if batch_i % batch_per_lot == batch_per_lot - 1:
                model.divide_clip_grads()
                model.add_noise_to_grads()
                optimizer.step()
            elif (party == 'casia' and batch_i == len(train_loader_casia) - 1) or \
                (party == 'hit' and batch_i == len(train_loader_hit) - 1) or \
                (party == 'combined' and batch_i == len(train_loader_combined) - 1):  # reach the end of the last incomplete lot
                model.divide_clip_grads(batch_i % batch_per_lot + 1)
                model.add_noise_to_grads(batch_i % batch_per_lot + 1)
                optimizer.step()
        else:
            optimizer.step()
    
    if args.setting in ['fedavg', 'casia']:
        for batch_i, (data, target) in enumerate(train_loader_casia):  # SWITCH OFF
            update(data, target, models[0], optims[0], 'casia', batch_i, batches_per_lot_casia)
    if args.setting in ['fedavg', 'hit']:
        for batch_i, (data, target) in enumerate(train_loader_hit):  # SWITCH OFF
            update(data, target, models[1], optims[1], 'hit', batch_i, batches_per_lot_hit)
    if args.setting == 'combined':
        for batch_i, (data, target) in enumerate(train_loader_combined):  # SWITCH OFF
            update(data, target, models[2], optims[2], 'combined', batch_i, batches_per_lot_combined)
            
#     for batch_i, (data, target) in enumerate(remote_loader_casia):  # SWITCH ON
#         update(data, target, models[0], optims[0], 'casia')
#     for batch_i, (data, target) in enumerate(remote_loader_hit):  # SWITCH ON
#         update(data, target, models[1], optims[1], 'hit')
    
    loss_casia, loss_hit, loss_combined = losses[0], losses[1], losses[2]
    correct_casia, correct_hit, correct_combined = corrects[0], corrects[1], corrects[2]
    
    loss_casia /= len(trainset_casia)
    loss_hit /= len(trainset_hit)
    loss_combined /= len(trainset_combined)
    acc_casia = correct_casia / len(trainset_casia)
    acc_hit = correct_hit / len(trainset_hit)
    acc_combined = correct_combined / len(trainset_combined)
    if args.setting in ['casia', 'hit', 'fedavg']:
        print('Trn loss_casia {:.4f}, loss_hit {:.4f}, acc_casia {:.4f}, acc_hit {:.4f}'.format(loss_casia, loss_hit, acc_casia, acc_hit))
    elif args.setting == 'combined':
        print('Trn loss_combined {:.4f}, acc_combined {:.4f}'.format(loss_combined, acc_combined))
    
    if args.setting == 'casia':
        with torch.no_grad():
            for p1, p2 in zip(models[0].parameters(), models[1].parameters()):
                p2.set_(p1.data)

    elif args.setting == 'hit':
        with torch.no_grad():
            for p1, p2 in zip(models[0].parameters(), models[1].parameters()):
                p1.set_(p2.data)

    elif args.setting == 'fedavg' and (epoch % args.freq == args.freq - 1 or epoch == args.epochs - 1):
        if args.freq > 1:
            print('Fedavg now')
        if args.spdz:
            new_params = list()
            for param_i in range(len(params[0])):
                spdz_params = list()
                spdz_params.append(params[0][param_i].copy().cpu().fix_precision().share(casia, hit, crypto_provider=crypto))
                spdz_params.append(params[1][param_i].copy().cpu().fix_precision().share(casia, hit, crypto_provider=crypto))
        #         if str(device) == 'cpu':  SWITCH ON
                      # see https://github.com/OpenMined/PySyft/pull/2990
        #             spdz_params.append(params[0][param_i].copy().get().fix_precision().share(casia, hit, crypto_provider=crypto))
        #             spdz_params.append(params[1][param_i].copy().get().fix_precision().share(casia, hit, crypto_provider=crypto))
        #         else:
        #             spdz_params.append(params[0][param_i].copy().cpu().get().fix_precision().share(casia, hit, crypto_provider=crypto))
        #             spdz_params.append(params[1][param_i].copy().cpu().get().fix_precision().share(casia, hit, crypto_provider=crypto))

                new_param = (spdz_params[0] * ratio_casia + spdz_params[1] * ratio_hit).get().float_precision() / (ratio_casia + ratio_hit)
                new_params.append(new_param)

            with torch.no_grad():
                for model in params:
                    for param in model:
                        param *= 0

        #         for model in models:  # SWITCH ON
        #             model.get()

                for param_index in range(len(params[0])):
                    if str(device) == 'cpu':
                        params[0][param_index].set_(new_params[param_index])
                        params[1][param_index].set_(new_params[param_index])
                    else:
                        params[0][param_index].set_(new_params[param_index].cuda())
                        params[1][param_index].set_(new_params[param_index].cuda())
        else:
            with torch.no_grad():
                for p1, p2 in zip(models[0].parameters(), models[1].parameters()):
                    p1.set_((p1.data * ratio_casia + p2.data * ratio_hit) / (ratio_casia + ratio_hit))
                    p2.set_(p1.data)

def val(loader_casia, loader_hit, loader_combined=None):
    
    if args.freq == 1:
        assert len(params[0]) == len(params[1])
        for param_index in range(len(params[0])):
            assert torch.equal(params[0][param_index], params[1][param_index])
    
#     model_casia.eval()  # doesn't work right
    losses = [0, 0, 0]
    corrects = [0, 0, 0]
    
    def val_batch(data, target, model, party):
        assert party in ['casia', 'hit', 'combined']
        data, target = data.to(device), target.to(device)  # dev
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)    # get the index of the max log-probability
        if party == 'casia':
            corrects[0] += pred.eq(target.view_as(pred)).sum().item()
            losses[0] += F.nll_loss(output, target, reduction='sum').item()
        elif party == 'hit':
            corrects[1] += pred.eq(target.view_as(pred)).sum().item()
            losses[1] += F.nll_loss(output, target, reduction='sum').item()
        elif party == 'combined':
            corrects[2] += pred.eq(target.view_as(pred)).sum().item()
            losses[2] += F.nll_loss(output, target, reduction='sum').item()
    
    if args.setting in ['casia', 'hit', 'fedavg']: 
        for data, target in loader_casia:
            val_batch(data, target, models[0], 'casia')
        for data, target in loader_hit:
            val_batch(data, target, models[1], 'hit')
    elif args.setting == 'combined':
        for data, target in loader_combined:
            val_batch(data, target, models[2], 'combined')
    
    loss_casia, loss_hit, loss_combined = losses[0], losses[1], losses[2]
    correct_casia, correct_hit, correct_combined = corrects[0], corrects[1], corrects[2]

    loss_casia /= len(loader_casia.dataset)
    loss_hit /= len(loader_hit.dataset)
    loss_combined /= len(loader_combined.dataset)
    acc_casia = correct_casia / len(loader_casia.dataset)
    acc_hit = correct_hit / len(loader_hit.dataset)
    acc_combined = correct_combined / len(loader_combined.dataset)
    
    if args.setting in ['casia', 'hit', 'fedavg']:
        print('Evaluated loss_casia {:.4f}, loss_hit {:.4f}, acc_casia {:.4f}, acc_hit {:.4f}'.format(loss_casia, loss_hit, acc_casia, acc_hit))
    elif args.setting == 'combined':
        print('Evaluated loss_combined {:.4f}, acc_combined {:.4f}'.format(loss_combined, acc_combined))

# Train
for epoch in range(args.starting_epoch, args.epochs):
    print('Epoch', epoch)
    t1 = int(time.time())
    train(epoch)
    t2 = int(time.time())
    val(val_loader_casia, val_loader_hit, val_loader_combined)
    if epoch % 5 == 4:
        print('Testing now')
        val(tst_loader_casia, tst_loader_hit, tst_loader_combined)
    t3 = int(time.time())
    print('Epoch trn time {:d}s, val time {:d}s'.format(t2-t1, t3-t2))
#     torch.save(models[0].state_dict(), "models/fl-hit-casia-{}.pt".format(epoch))

# val(tst_loader_casia, tst_loader_hit)