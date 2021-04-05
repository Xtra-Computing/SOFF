from data import *
import torch
import numpy as np
import sys
import argparse
import PIL
import time
import os
import errno
import threading
import math
from scipy.optimize import newton, bisect
from torch import nn
from torchvision import models, transforms
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

## Parse arguments #############################################################
argv = sys.argv


def printHelp():
    print("Usage: {}".format(argv[0]))


available_datasets = ['imdbwiki', 'allagefaces', 'appa']

parser = argparse.ArgumentParser(
    description='train with fedavg and dp',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-E',
                    '--epochs',
                    default=50,
                    type=int,
                    help="Number of epochs")
parser.add_argument('-d',
                    '--data-dir',
                    default="preprocessed_data",
                    help="The directory storing preprocessed data")
parser.add_argument('-b',
                    '--batch-size',
                    default=50,
                    type=int,
                    help="How many record per batch")
parser.add_argument('-a',
                    '--average-every',
                    default=1,
                    type=int,
                    help="How many epochs per averaging")
parser.add_argument('-s',
                    '--save-dir',
                    default="saves",
                    type=str,
                    help="Model save dir")
parser.add_argument(
    '-l',
    '--load-model',
    type=str,
    help="Load previous save point, must also specify starting epoch")
parser.add_argument(
    '-ee',
    '--ending-epoch',
    type=int,
    help="Last ending epoch number. Used for continuing from last save")
parser.add_argument('-ds',
                    '--datasets',
                    default=['allagefaces', 'appa'],
                    type=str,
                    nargs='+',
                    help='|'.join(available_datasets))
parser.add_argument('-sc',
                    '--scatter',
                    action="store_true",
                    help="Scatter to multple GPUs")
parser.add_argument('-ep',
                    '--epsilon',
                    type=float,
                    default=2.0,
                    help="epsilon value for DP")
parser.add_argument('-ls',
                    '--lotsize-scaler',
                    default=1.0,
                    type=float,
                    help="Scale the lot size sqrt(N) by a multiplier")
parser.add_argument('-n', '--no-noise', action='store_true')
args = parser.parse_args()

if bool(args.load_model is None) != bool(args.ending_epoch is None):
    print("-l and -ee must be specified at the same time")
    exit(1)

for dataset in args.datasets:
    if not dataset in available_datasets:
        print("Unrecognized dataset")
        exit(1)

print(args)

devices = []
if args.scatter:
    for i in range(torch.cuda.device_count()):
        devices.append(torch.device('cuda:{}'.format(i)))
else:
    devices.append(torch.device('cuda:0'))

## Intialize data loader #######################################################
print("Loading data...")

age_classes = torch.arange(101)
datasets = []
for dataset in args.datasets:
    if dataset == 'imdbwiki':
        datasets.append(Dataset_ImdbWiki(args.data_dir))
    elif dataset == 'appa':
        datasets.append(Dataset_APPA(args.data_dir))
    elif dataset == 'allagefaces':
        datasets.append(Dataset_AllAgeFaces(args.data_dir))
    else:
        print("{} not implemented yet".format(dataset))
        exit(1)

datasets_splitted = []
for dataset in datasets:
    train_size = int(len(dataset) * 0.9)
    val_size = int((len(dataset) - train_size) * 0.5)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])
    datasets_splitted.append({
        'train': train_dataset,
        'test': test_dataset,
        'val': val_dataset
    })


class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.features = torch.stack(transposed_data[0], 0)
        self.labels = torch.Tensor(transposed_data[1]).round().long()

    def pin_memory(self):
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        return self


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


dataloaders = []
for dataset in datasets_splitted:
    dataloaders.append({
        'train':
        torch.utils.data.DataLoader(dataset['train'],
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    collate_fn=collate_wrapper),
        'test':
        torch.utils.data.DataLoader(dataset['test'],
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    collate_fn=collate_wrapper),
        'val':
        torch.utils.data.DataLoader(dataset['val'],
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    collate_fn=collate_wrapper)
    })

# For testing...
# import matplotlib.pyplot as plt
# NUM_IMAGE=4
# for batch in dataloaders['train']:
#     for i, sample in enumerate(batch.features):
#         if i >= NUM_IMAGE:
#             break
#         print(i, sample.shape)
#         ax = plt.subplot(1, NUM_IMAGE, i+1)
#         plt.tight_layout()
#         ax.set_title('age: {:.2f}'.format(batch.labels[i]))
#         ax.axis('off')
#         plt.imshow(sample.numpy().transpose(1,2,0))
#     break
# plt.show()
# exit(1)

## DP Config ###################################################################

clip = 10  # TODO: decide clip value

## Deprecated
#
# lotSize = args.batches_per_lot * args.batch_size# L
# delta = 10**(-5)
#
# assert(trainXImdb.shape[0] == trainXAmazon.shape[0]) # otherwise we need two qs and Ts
# lotsPerEpoch = trainXImdb.shape[0] / lotSize
# q = lotSize / trainXImdb.shape[0]
# T = args.epochs * lotsPerEpoch
#
# # sigma = np.sqrt(2 * np.log(1.25/delta))/args.epsilon # this is the sigma of strong composition
# sigma = 2 * q * np.sqrt(T * np.log(1./delta)) / args.epsilon # this is the sigma of moment accountant

lotsizes = [
    len(dataset['train'])**.5 * args.lotsize_scaler
    for dataset in datasets_splitted
]
batches_per_lot_list = [
    max(1, round(lotsize / args.batch_size)) for lotsize in lotsizes
]
lotsizes = [
    args.batch_size * batches_per_lot_list[i]
    for i, dataset in enumerate(datasets_splitted)
]

deltas = [
    min(10**(-5), 1 / len(dataset['train'])) for dataset in datasets_splitted
]

lots_per_epoch_list = [
    math.ceil(len(dataset['train']) / lotsizes[i])
    for i, dataset in enumerate(datasets_splitted)
]


def compute_dp_sgd_wrapper_generator(i):
    def compute_dp_sgd_wrapper(_sigma):
        return compute_dp_sgd_privacy.compute_dp_sgd_privacy(
            n=len(datasets_splitted[i]['train']),
            batch_size=lotsizes[i],
            noise_multiplier=_sigma,
            epochs=args.epochs,
            delta=deltas[i])[0] - args.epsilon

    return compute_dp_sgd_wrapper


sigmas = [
    bisect(compute_dp_sgd_wrapper_generator(i), 0.01, 10000)
    for i, _ in enumerate(lotsizes)
]

print('--> bpls{} = {}'.format(list(range(len(batches_per_lot_list))), batches_per_lot_list))
print('--> sigmas{} = {}'.format(list(range(len(sigmas))), sigmas))
print('--> actual epslion{} = {}'.format(list(range(len(sigmas))), [
    compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=len(datasets_splitted[i]['train']),
        batch_size=lotsizes[i],
        noise_multiplier=sigmas[i],
        epochs=args.epochs,
        delta=deltas[i])[0] for i, _ in enumerate(lotsizes)
]))

## Initialize network ##########################################################
print("Initializing network...")

nets = []
for i in range(len(datasets)):
    nets.append(models.vgg16(pretrained=True))
    nets[-1].classifier[6] = nn.Linear(4096, 101)

if args.load_model:
    print("Loading saved model...")
    for net in nets:
        net.load_state_dict(torch.load(args.load_model))

for i, net in enumerate(nets):
    net.cuda(devices[i % len(devices)])

# Noise adding utility
_noisesToAdd = {}


def _gaussian_noise(grad, sigma):
    global _noisesToAdd
    if not tuple(grad.shape) in _noisesToAdd:
        _noisesToAdd[tuple(grad.shape)] = torch.zeros(grad.shape).cuda(
            devices[-1])
    _noisesToAdd[tuple(grad.shape)].data.normal_(0.0, std=sigma * clip)
    return _noisesToAdd[tuple(grad.shape)]


# add noise to the i-th net
def add_gaussian_noise(net_idx):
    for key, param in nets[net_idx].named_parameters():
        noise = 1 / lotsizes[net_idx] * _gaussian_noise(
            param.grad, sigmas[net_idx])
        param.grad += noise.cuda(devices[net_idx % len(devices)])


#add_gaussian_noise(0)
#print(_noisesToAdd)
#exit(1)


# defined prediction network
class Prediction(nn.Module):
    def __init__(self, device):
        super(Prediction, self).__init__()
        self.softmax = nn.Softmax(1)
        self.ages = torch.arange(101).cuda(device)

    def forward(self, x):
        return torch.sum(self.softmax(x) * self.ages, 1)


preds = []
for i in range(len(datasets)):
    preds.append(Prediction(devices[i % len(devices)]))
    preds[-1].cuda(devices[i % len(devices)])
    preds[-1].eval()  # prediction network never rerquire gard

learningRate = 0.0001

optimizers = []
for net in nets:
    optimizers.append(
        torch.optim.SGD([
            {
                'params': net.features.parameters()
            },
            {
                'params': net.avgpool.parameters()
            },
            {
                'params': net.classifier[:6].parameters()
            },
            {
                'params': net.classifier[6].parameters(),
                'lr': 0.001
            },
        ],
                        lr=learningRate,
                        momentum=0.9,
                        weight_decay=0.0005))

# resnet
# param_list = []
# last_layer = ['fc.weight', 'fc.bias']
# for name, param in net.named_parameters():
#     if not name in last_layer:
#         param_list.append({ 'params': param })
#     else:
#         param_list.append({ 'params': param, 'lr': 0.001})
# optimizer = torch.optim.SGD(param_list, lr=learningRate, momentum=0.9, weight_decay=0.0005)

lr_schedulers = []
for optimizer in optimizers:
    lr_schedulers.append(
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1))

train_criterion = nn.CrossEntropyLoss()
eval_criterion_mae = nn.L1Loss()

## Train #######################################################################
try:
    os.mkdir(args.save_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

print("Start training...")

starting_epoch = 0
if args.ending_epoch:
    print("Continuing from finished epoch {}".format(args.ending_epoch))
    starting_epoch = args.ending_epoch
    for i in range(args.ending_epoch):
        for schedular in lr_schedulers:
            schedular.step()

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    print_lock.acquire()
    print(*args, **kwargs)
    print_lock.release()


# cross thread communcation events,
# starting_events: the aggregation has been done. thread[i] now may start.
# ending_events: thread[i] has ended calculating one epoch. The main thread can continue to aggregate the weights from all models once all threads has ended
starting_events = []
ending_events = []
for i in range(len(devices)):
    starting_events.append(threading.Event())
    ending_events.append(threading.Event())


def training_on_one_gpu(gpu_idx):
    for e in range(starting_epoch, starting_epoch + args.epochs):
        starting_events[gpu_idx].wait()
        starting_events[gpu_idx].clear()

        safe_print("Thread {}:Epoch {}/{}".format(
            gpu_idx, e + 1, starting_epoch + args.epochs))

        for data_idx in filter(lambda x: x % len(devices) == gpu_idx,
                               range(len(datasets))):
            safe_print("Thread {} ds {}:Training on datastet {}: {}".format(
                gpu_idx, data_idx, data_idx, args.datasets[data_idx]))

            # train ####################################################################
            nets[data_idx].train()
            counter = 0

            losses_train = []
            losses_train_mae = []
            acc_train = []
            acc_train_one_off = []
            for data in dataloaders[data_idx]['train']:
                counter += 1
                safe_print("Thread {} ds {}:  Step:     {}/{}".format(
                    gpu_idx, data_idx, counter * args.batch_size,
                    len(dataloaders[data_idx]['train']) * args.batch_size))

                features = data.features.cuda(devices[gpu_idx])
                labels = data.labels.cuda(devices[gpu_idx])

                with torch.set_grad_enabled(True):
                    prediction = nets[data_idx](features)
                    loss = train_criterion(prediction, labels)
                    loss.backward()

                    if counter % batches_per_lot_list[
                            data_idx] == 0 or counter == math.ceil(
                                len(dataloaders[data_idx]['train']) /
                                args.batch_size):
                        nn.utils.clip_grad_norm_(nets[data_idx].parameters(),
                                                 clip)
                        for _, param in nets[data_idx].named_parameters():
                            param.grad /= batches_per_lot_list[data_idx]
                        if not args.no_noise:
                            safe_print("Thread {} ds {}: Adding noise...".format(gpu_idx, data_idx))
                            add_gaussian_noise(net_idx=data_idx)

                        optimizers[data_idx].step()
                        optimizers[data_idx].zero_grad()

                # go through the prediction steps
                age_prediction = preds[data_idx](prediction)

                # calculate different types of losses
                losses_train.append(loss.item())
                losses_train_mae.append(
                    eval_criterion_mae(age_prediction, labels.float()).item())
                acc_train.append(
                    (labels == age_prediction.round()).float().sum().item() /
                    len(data.features))
                acc_train_one_off.append(
                    (acc_train[-1] +
                     (labels + 1
                      == age_prediction.round()).float().sum().item() +
                     (labels - 1
                      == age_prediction.round()).float().sum().item()) /
                    len(data.features))

                if counter % max(batches_per_lot_list[data_idx], 50) == 0:
                    print_lock.acquire()
                    print("Thread {} ds {}:".format(gpu_idx, data_idx))
                    print("Thread {} ds {}:    Train Loss:      {}".format(
                        gpu_idx, data_idx, np.average(losses_train)))
                    print("Thread {} ds {}:    Train MAE:       {}".format(
                        gpu_idx, data_idx, np.average(losses_train_mae)))
                    print("Thread {} ds {}:    Train Acc:       {}".format(
                        gpu_idx, data_idx, np.average(acc_train)))
                    print("Thread {} ds {}:    Train 1-off Acc: {}".format(
                        gpu_idx, data_idx, np.average(acc_train_one_off)))
                    print_lock.release()

                    losses_train = []
                    losses_train_mae = []
                    acc_train = []
                    acc_train_one_off = []

            lr_schedulers[data_idx].step()

            # eval #####################################################################
            safe_print("Thread {} ds {}:  Eval".format(gpu_idx, data_idx))
            nets[data_idx].eval()

            losses_eval = []
            losses_eval_mae = []
            acc_eval = []
            acc_eval_one_off = []

            counter = 0
            for data in dataloaders[data_idx]['val']:
                counter += 1
                safe_print("Thread {} ds {}:  Step:     {}/{}".format(
                    gpu_idx, data_idx, counter,
                    len(dataloaders[data_idx]['val'])))
                features = data.features.cuda(devices[gpu_idx])
                labels = data.labels.cuda(devices[gpu_idx])
                with torch.set_grad_enabled(False):
                    prediction = nets[data_idx](features)
                    loss = train_criterion(prediction, labels)

                # go through the prediction steps
                age_prediction = preds[data_idx](prediction)

                # calculate different types of losses
                losses_eval.append(loss.item())
                losses_eval_mae.append(
                    eval_criterion_mae(age_prediction, labels.float()).item())
                acc_eval.append(
                    (labels == age_prediction.round()).float().sum().item() /
                    len(data.features))
                acc_eval_one_off.append(
                    (acc_eval[-1] +
                     (labels + 1
                      == age_prediction.round()).float().sum().item() +
                     (labels - 1
                      == age_prediction.round()).float().sum().item()) /
                    len(data.features))

            print_lock.acquire()
            print("Thread {} ds {}:    Val Loss:        {}".format(
                gpu_idx, data_idx, np.average(losses_eval)))
            print("Thread {} ds {}:    Val MAE:         {}".format(
                gpu_idx, data_idx, np.average(losses_eval_mae)))
            print("Thread {} ds {}:    Val Acc:         {}".format(
                gpu_idx, data_idx, np.average(acc_eval)))
            print("Thread {} ds {}:    Val 1-off Acc:   {}".format(
                gpu_idx, data_idx, np.average(acc_eval_one_off)))
            print_lock.release()
        #- for data_idx in filter(range(datastes))

        ending_events[gpu_idx].set()
    #- for e in range(epochs)


try:
    threads = []
    for i in range(len(devices)):
        threads.append(
            threading.Thread(target=training_on_one_gpu, kwargs={'gpu_idx':
                                                                 i}))
        threads[-1].start()
except:
    print("Error: unable to start thread")

total_num_data = sum([len(dataset['train']) for dataset in datasets_splitted])
ratios = [
    len(dataset['train']) / total_num_data for dataset in datasets_splitted
]

losses_eval_epochs = []
losses_eval_mae_epochs = []
for e in range(starting_epoch, starting_epoch + args.epochs):
    safe_print("Thread main: Epoch {}/{}".format(e + 1,
                                                 starting_epoch + args.epochs))
    # Now we start the main thread
    for event in starting_events:
        event.set()

    # wait for all threads to end
    for event in ending_events:
        event.wait()
        event.clear()

    # do aggregation here, in-place modification to save memory
    safe_print("Thread main: Strating aggregation")
    total_num_data = sum(
        [len(dataset['train']) for dataset in datasets_splitted])
    ratios = [
        len(dataset['train']) / total_num_data for dataset in datasets_splitted
    ]
    for i, net in enumerate(nets):
        for name, param in net.named_parameters():
            param.data.copy_(param.data * ratios[i])

    for net in nets[1:]:
        for name, param in nets[0].named_parameters():
            param.data.copy_(
                param.data +
                dict(net.named_parameters())[name].data.cuda(devices[0]))

    for i, net in enumerate(nets[1:]):
        for name, param in net.named_parameters():
            param.data.copy_(
                dict(nets[0].named_parameters())[name].data.cuda(
                    devices[(i + 1) % len(devices)]))

    # do evaluation to the aggregated model
    safe_print("Thread main: Eval")

    # eval #####################################################################
    safe_print("Thread main:  Eval".format(nets[0]))

    losses_eval = []
    losses_eval_mae = []
    acc_eval = []
    acc_eval_one_off = []

    for data_idx in range(len(datasets)):
        counter = 0
        for data in dataloaders[data_idx]['val']:
            counter += 1
            safe_print("Thread main ds {}:  Step:     {}/{}".format(
                data_idx, counter, len(dataloaders[data_idx]['val'])))
            features = data.features.cuda(devices[0])
            labels = data.labels.cuda(devices[0])
            with torch.set_grad_enabled(False):
                prediction = nets[0](features)
                loss = train_criterion(prediction, labels)

            # go through the prediction steps
            age_prediction = preds[0](prediction)

            # calculate different types of losses
            losses_eval.append(loss.item())
            losses_eval_mae.append(
                eval_criterion_mae(age_prediction, labels.float()).item())
            acc_eval.append(
                (labels == age_prediction.round()).float().sum().item() /
                len(data.features))
            acc_eval_one_off.append(
                (acc_eval[-1] +
                 (labels + 1 == age_prediction.round()).float().sum().item() +
                 (labels - 1 == age_prediction.round()).float().sum().item()) /
                len(data.features))

    print_lock.acquire()
    print("Thread main:    Val Loss:        {}".format(
        np.average(losses_eval)))
    print("Thread main:    Val MAE:         {}".format(
        np.average(losses_eval_mae)))
    print("Thread main:    Val Acc:         {}".format(np.average(acc_eval)))
    print("Thread main:    Val 1-off Acc:   {}".format(
        np.average(acc_eval_one_off)))
    print_lock.release()

    # save model every if current model is better than all previous models
    if len(losses_eval_epochs
           ) == 0 or min(losses_eval_epochs) >= np.average(losses_eval):
        torch.save(
            nets[0].state_dict(),
            args.save_dir + "/model_fedavg_minloss_{}{}".format(
                '_'.join(args.datasets),
                "_withpretrain" if args.load_model else ""))
        with open(
                args.save_dir +
                "/model_fedavg_minloss_{}{}_details.txt".format(
                    '_'.join(args.datasets),
                    "_withpretrain" if args.load_model else ""), "w") as f:
            f.write("Trainig epoch of the saved model: {}".format(e + 1))
    losses_eval_epochs.append(np.average(losses_eval))

    # save model every if current model is better than all previous models (by MAE)
    if len(losses_eval_mae_epochs) == 0 or min(
            losses_eval_mae_epochs) >= np.average(losses_eval_mae):
        torch.save(
            nets[0].state_dict(),
            args.save_dir + "/model_fedavg_minmae_{}{}".format(
                '_'.join(args.datasets),
                "_withpretrain" if args.load_model else ""))
        with open(
                args.save_dir + "/model_fedavg_minmae_{}{}_details.txt".format(
                    '_'.join(args.datasets),
                    "_withpretrain" if args.load_model else ""), "w") as f:
            f.write("Trainig epoch of the saved mode: {}".format(e + 1))
    losses_eval_mae_epochs.append(np.average(losses_eval_mae))

    # save the last model
    if e == (starting_epoch + args.epochs - 1):
        torch.save(
            nets[0].state_dict(),
            args.save_dir + "/model_fedavg_last_{}{}".format(
                '_'.join(args.datasets),
                "_withpretrain" if args.load_model else ""))
