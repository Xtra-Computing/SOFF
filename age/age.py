from data import *
import torch
import numpy as np
import sys
import argparse
import PIL
import time
import os
import errno
from torch import nn
from torchvision import models, transforms

## Parse arguments #############################################################
argv = sys.argv


def printHelp():
    print("Usage: {}".format(argv[0]))


available_datasets = [
    'imdbwiki', 'allagefaces', 'appa', 'allagefaces_appa', 'all'
]

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
    help="Load previous save point, must also specify ending epoch")
parser.add_argument(
    '-ee',
    '--ending-epoch',
    type=int,
    help="Last ending epoch number. Used for continuing from last save")
parser.add_argument('-ds',
                    '--dataset',
                    default='imdbwiki',
                    type=str,
                    help='|'.join(available_datasets))
args = parser.parse_args()

if bool(args.load_model is None) != bool(args.ending_epoch is None):
    print("-l and -ee must be specified at the same time")
    exit(1)

if not args.dataset in available_datasets:
    print("Unrecognized dataset")
    exit(1)

print(args)

## Intialize data loader #######################################################
print("Loading data...")

age_classes = torch.arange(101)
if args.dataset == 'imdbwiki':
    dataset = Dataset_ImdbWiki(args.data_dir)
elif args.dataset == 'allagefaces':
    dataset = Dataset_AllAgeFaces(args.data_dir)
elif args.dataset == 'appa':
    dataset = Dataset_APPA(args.data_dir)
elif args.dataset == 'allagefaces_appa':
    dataset = Dataset_AllAgeFaces_APPA(args.data_dir)
else:
    print("not implemented yet")
    exit(1)

test_datasets = {}
for ds in ['allagefaces', 'appa', 'allagefaces_appa']:
    if ds == 'imdbwiki':
        dataset = Dataset_ImdbWiki(args.data_dir)
    elif ds == 'allagefaces':
        dataset = Dataset_AllAgeFaces(args.data_dir)
    elif ds == 'appa':
        dataset = Dataset_APPA(args.data_dir)
    elif ds == 'allagefaces_appa':
        dataset = Dataset_AllAgeFaces_APPA(args.data_dir)
    else:
        print("not implemented yet")
        exit(1)

    _train_size = int(len(dataset) * 0.9)
    _val_size = int((len(dataset) - _train_size) * 0.5)
    _test_size = len(dataset) - _train_size - _val_size

    _train_dataset, _val_dataset, _test_dataset = torch.utils.data.random_split(
        dataset, [_train_size, _val_size, _test_size])

    test_datasets[ds] = _test_dataset

    if args.dataset == ds:
        train_dataset = _train_dataset
        val_dataset = _val_dataset


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


dataloaders = {
    'train':
    torch.utils.data.DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                collate_fn=collate_wrapper),
    'test': {
        'allagefaces':
        torch.utils.data.DataLoader(test_datasets['allagefaces'],
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    collate_fn=collate_wrapper),
        'appa':
        torch.utils.data.DataLoader(test_datasets['appa'],
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    collate_fn=collate_wrapper),
        'allagefaces_appa':
        torch.utils.data.DataLoader(test_datasets['allagefaces_appa'],
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    collate_fn=collate_wrapper),
    },
    'val':
    torch.utils.data.DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                collate_fn=collate_wrapper)
}

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

## Initialize network ##########################################################
print("Initializing network...")

net = models.vgg16(pretrained=True)
net.classifier[6] = nn.Linear(4096, 101)
if args.load_model:
    print("Loading saved model...")
    net.load_state_dict(torch.load(args.load_model))
net.cuda()

# defined prediction network


class Prediction(nn.Module):
    def __init__(self):
        super(Prediction, self).__init__()
        self.softmax = nn.Softmax(1)
        self.ages = torch.arange(101).cuda()

    def forward(self, x):
        return torch.sum(self.softmax(x) * self.ages, 1)


pred = Prediction()
pred.cuda()
pred.eval()  # prediction network never rerquire gard

learningRate = 0.0001

optimizer = torch.optim.SGD([
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
                            weight_decay=0.0005)

# resnet
# param_list = []
# last_layer = ['fc.weight', 'fc.bias']
# for name, param in net.named_parameters():
#     if not name in last_layer:
#         param_list.append({ 'params': param })
#     else:
#         param_list.append({ 'params': param, 'lr': 0.001})
# optimizer = torch.optim.SGD(param_list, lr=learningRate, momentum=0.9, weight_decay=0.0005)
lrSchedular = torch.optim.lr_scheduler.StepLR(optimizer,
                                              step_size=10,
                                              gamma=0.1)

train_criterion = nn.CrossEntropyLoss()
eval_criterion_mae = nn.L1Loss()

## Train #######################################################################
try:
    os.mkdir(args.save_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

print("Start training...")
losses_eval_epochs = []
losses_eval_mae_epochs = []

starting_epoch = 0
if args.ending_epoch:
    print("Continuing from finished epoch {}".format(args.ending_epoch))
    starting_epoch = args.ending_epoch
    for i in range(args.ending_epoch):
        lrSchedular.step()

for e in range(starting_epoch, starting_epoch + args.epochs):
    # train ####################################################################
    net.train()
    counter = 0
    print("Epoch {}/{}".format(e + 1, starting_epoch + args.epochs))

    losses_train = []
    losses_train_mae = []
    acc_train = []
    acc_train_one_off = []
    for data in dataloaders['train']:
        counter += 1
        print("\r  Step:     {}/{}".format(
            counter * args.batch_size,
            len(dataloaders['train']) * args.batch_size),
              end="")

        features = data.features.cuda(non_blocking=True)
        labels = data.labels.cuda(non_blocking=True)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            prediction = net(features)
            loss = train_criterion(prediction, labels)
            loss.backward()
            optimizer.step()

        # go through the prediction steps
        age_prediction = pred(prediction)

        # calculate different types of losses
        losses_train.append(loss.item())
        losses_train_mae.append(
            eval_criterion_mae(age_prediction, labels.float()).item())
        acc_train.append(
            (labels == age_prediction.round()).float().sum().item() /
            len(data.features))
        acc_train_one_off.append(
            (acc_train[-1] +
             (labels + 1 == age_prediction.round()).float().sum().item() +
             (labels - 1 == age_prediction.round()).float().sum().item()) /
            len(data.features))

        if counter % 100 == 0:
            print("")
            print("    Train Loss:      {}".format(np.average(losses_train)))
            print("    Train MAE:       {}".format(
                np.average(losses_train_mae)))
            print("    Train Acc:       {}".format(np.average(acc_train)))
            print("    Train 1-off Acc: {}".format(
                np.average(acc_train_one_off)))
            losses_train = []
            losses_train_mae = []
            acc_train = []
            acc_train_one_off = []

    lrSchedular.step()

    # eval #####################################################################
    print("")
    print("  Eval")
    net.eval()

    losses_eval = []
    losses_eval_mae = []
    acc_eval = []
    acc_eval_one_off = []

    counter = 0
    for data in dataloaders['val']:
        counter += 1
        print("\r  Step:     {}/{}".format(counter, len(dataloaders['val'])),
              end="")
        features = data.features.cuda()
        labels = data.labels.cuda()
        with torch.set_grad_enabled(False):
            prediction = net(features)
            loss = train_criterion(prediction, labels)

        # go through the prediction steps
        age_prediction = pred(prediction)

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

    print("")
    print("    Val Loss:        {}".format(np.average(losses_eval)))
    print("    Val MAE:         {}".format(np.average(losses_eval_mae)))
    print("    Val Acc:         {}".format(np.average(acc_eval)))
    print("    Val 1-off Acc:   {}".format(np.average(acc_eval_one_off)))

    # save model every if current model is better than all previous models
    if len(losses_eval_epochs
           ) == 0 or min(losses_eval_epochs) >= np.average(losses_eval):
        torch.save(
            net.state_dict(), args.save_dir + "/model_minloss_{}{}".format(
                args.dataset, "_withpretrain" if args.load_model else ""))
        with open(
                args.save_dir + "/model_minloss_{}{}_details.txt".format(
                    args.dataset, "_withpretrain" if args.load_model else ""),
                "w") as f:
            f.write("Trainig epoch of the saved model: {}".format(e + 1))
    losses_eval_epochs.append(np.average(losses_eval))

    # save model every if current model is better than all previous models (by MAE)
    if len(losses_eval_mae_epochs) == 0 or min(
            losses_eval_mae_epochs) >= np.average(losses_eval_mae):
        torch.save(
            net.state_dict(), args.save_dir + "/model_minmae_{}{}".format(
                args.dataset, "_withpretrain" if args.load_model else ""))
        with open(
                args.save_dir + "/model_minmae_{}{}_details.txt".format(
                    args.dataset, "_withpretrain" if args.load_model else ""),
                "w") as f:
            f.write("Trainig epoch of the saved mode: {}".format(e + 1))
    losses_eval_mae_epochs.append(np.average(losses_eval_mae))

    # save the last model
    if e == (starting_epoch + args.epochs - 1):
        torch.save(
            net.state_dict(), args.save_dir + "/model_last_{}{}".format(
                args.dataset, "_withpretrain" if args.load_model else ""))

    # test #####################################################################

    counter = 0
    for ds_name in dataloaders['test'].keys():
        losses_test = []
        losses_test_mae = []
        acc_test = []
        acc_test_one_off = []

        for data in dataloaders['test'][ds_name]:
            counter += 1
            print("\r  Step:     {}/{}".format(counter,
                                               len(dataloaders['val'])),
                  end="")
            features = data.features.cuda()
            labels = data.labels.cuda()
            with torch.set_grad_enabled(False):
                prediction = net(features)
                loss = train_criterion(prediction, labels)

            # go through the prediction steps
            age_prediction = pred(prediction)

            # calculate different types of losses
            losses_test.append(loss.item())
            losses_test_mae.append(
                eval_criterion_mae(age_prediction, labels.float()).item())
            acc_test.append(
                (labels == age_prediction.round()).float().sum().item() /
                len(data.features))
            acc_test_one_off.append(
                (acc_test[-1] +
                 (labels + 1 == age_prediction.round()).float().sum().item() +
                 (labels - 1 == age_prediction.round()).float().sum().item()) /
                len(data.features))

        print("")
        print("    Test Loss ({}):        {}".format(ds_name, np.average(losses_test)))
        print("    Test MAE ({}):         {}".format(ds_name, np.average(losses_test_mae)))
        print("    Test Acc ({}):         {}".format(ds_name, np.average(acc_test)))
        print("    Test 1-off Acc ({}):   {}".format(ds_name, np.average(acc_test_one_off)))
