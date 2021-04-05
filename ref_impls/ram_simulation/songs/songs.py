from data import *
import torch
import numpy as np
import sys
import argparse
import os
import errno
from torch import nn

## Parse arguments #############################################################
argv = sys.argv


def printHelp():
    print("Usage: {}".format(argv[0]))


available_datasets = ['msd', 'msd_aligned', 'fma', 'fma_aligned', 'aligned', 'union']

# fmt: off
parser = argparse.ArgumentParser(description='train a year prediction model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-E', '--epochs', default=1000, type=int, help="Number of epochs")
parser.add_argument('-d', '--data-dir', default="preprocessed_data", help="The directory storing preprocessed data")
parser.add_argument('-b', '--batch-size', default=256, type=int, help="How many record per batch")
parser.add_argument('-a', '--average-every', default=1, type=int, help="How many epochs per averaging")
parser.add_argument('-s', '--save-dir', default="saves", type=str, help="Model save dir")
parser.add_argument('-p', '--patience', default=30, type=int, help="LRSchedular patience")
parser.add_argument('-o', '--optimizer', default='sgd', type=str, help="sgd|adam")
parser.add_argument('-sn', '--splitnn', action='store_true', help="simulate splitNN, when using aligned or union dataset")
parser.add_argument('-lr', '--learning-rate', default=1e-7, type=float, help="Starting learning rate")
parser.add_argument('-ds', '--dataset', default='fma', type=str, help='|'.join(available_datasets))
args = parser.parse_args()
# fmt: on

if not args.dataset in available_datasets:
    raise Exception("Unrecognized dataset")
if args.splitnn and args.dataset not in ['aligned', 'union']:
    raise Exception("SplitNN should be used with federatd dataset (aligned|union)")

print(args)

## Initialize data loader #######################################################
print("Loading data...")

if args.dataset == 'msd':
    dataset = MSDDataset(args.data_dir)
elif args.dataset == 'msd_aligned':
    dataset = MSDAlignedDataset(args.data_dir)
elif args.dataset == 'fma':
    dataset = FMADataset(args.data_dir)
elif args.dataset == 'fma_aligned':
    dataset = FMAAlignedDataset(args.data_dir)
elif args.dataset == 'aligned':
    dataset = AlignedDataset(args.data_dir)
elif args.dataset == 'union':
    dataset = UnionDataset(args.data_dir)
else:
    raise Exception("No such dataset")

train_size = int(len(dataset) * 0.9)
val_size = int((len(dataset) - train_size) * 0.5)
test_size = len(dataset) - train_size - val_size

# we need to fix the splitting random seed so that the three aligned datasets use the same indices as test set
torch.manual_seed(42) # use this for torch < 1.6 compatibility
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size])
torch.manual_seed(torch.initial_seed())

dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True),
}

## Initialize network ##########################################################
print("Initializing network...")

# a simple regression network
if args.splitnn:
    class Net(nn.Module):
        def __init__(self, input_sizes, hidden_sizes_1, hidden_sizes_2):
            super().__init__()
            self.input_sizes = input_sizes
            self.hidden_sizes_1 = hidden_sizes_1
            self.hidden_sizes_2 = hidden_sizes_2

            self.l1s = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=True) for input_size, hidden_size in zip(input_sizes, hidden_sizes_1)])
            self.l2s = nn.ModuleList([nn.Linear(hidden_size_1, hidden_size_2, bias=True) for hidden_size_1, hidden_size_2 in zip(hidden_sizes_1, hidden_sizes_2)])
            self.output = nn.Linear(np.sum(hidden_sizes_2), 1)
            self.relu = nn.ReLU()
            pass
        def forward(self, x):
            xs = [l1(x[:,int(np.sum(self.input_sizes[0:i])):int(np.sum(self.input_sizes[0:i+1]))])
                    for i, l1 in enumerate(self.l1s)]
            xs = [self.relu(x) for x in xs]
            xs = [l2(x) for x, l2 in zip(xs, self.l2s)]
            xs = [self.relu(x) for x in xs]
            x = torch.cat(tuple(xs),dim=1)
            x = self.output(x)
            return x
    input_sizes = dataset.input_sizes()
    net = Net(input_sizes, [size * 2 for size in input_sizes], [size // 2 for size in input_sizes])

else:
    class Net(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2):
            super().__init__()
            self.l1 = nn.Linear(input_size, hidden_size_1, bias=True)
            self.l2 = nn.Linear(hidden_size_1, hidden_size_2, bias=True)
            self.output = nn.Linear(hidden_size_2, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.l1(x)
            x = self.relu(x)
            x = self.l2(x)
            x = self.relu(x)
            x = self.output(x)
            return x

    net = Net(dataset.input_size(), 2 * dataset.input_size(), dataset.input_size() // 2)

#if args.load_model:
#    print("Loading saved model...")
#    net.load_state_dict(torch.load(args.load_model))
net.cuda()

learningRate = args.learning_rate
learningRateFactor = 0.1

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(net.parameters(), lr=learningRate, momentum=0.9, )
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=learningRate)
else:
    raise Excecption("Unkown optimizer: {}".format(args.optimizer))
lrSchedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=learningRateFactor, patience=args.patience, verbose=True)

train_criterion = nn.MSELoss()
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

# TODO: load saved state dict
starting_epoch = 0
# if args.ending_epoch:
#     print("Continuing from finished epoch {}".format(args.ending_epoch))
#     starting_epoch = args.ending_epoch
#     for i in range(args.ending_epoch):
#         lrSchedular.step()
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
        if counter % 100 == 0:
            print("\r  Step:     {}/{}".format(counter * args.batch_size,
                                               len(dataloaders['train']) * args.batch_size), end="")
        counter += 1

        features = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            prediction = net(features)
            loss = train_criterion(prediction, labels)
            loss.backward()
            optimizer.step()

        # calculate different types of losses
        losses_train.append(loss.item())
        losses_train_mae.append(eval_criterion_mae(
            prediction, labels.float()).item())
        acc_train.append(
            (labels == prediction.round()).float().sum().item() / len(data[0]))
        acc_train_one_off.append(
            (acc_train[-1] +
             (labels + 1 == prediction.round()).float().sum().item() +
                (labels - 1 == prediction.round()).float().sum().item()) / len(data[0]))

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
        features = data[0].cuda()
        labels = data[1].cuda()
        with torch.set_grad_enabled(False):
            prediction = net(features)
            loss = train_criterion(prediction, labels)

        # calculate different types of losses
        losses_eval.append(loss.item())
        losses_eval_mae.append(
            eval_criterion_mae(prediction, labels.float()).item())
        acc_eval.append(
            (labels == prediction.round()).float().sum().item() /
            len(data[0]))
        acc_eval_one_off.append(
            (acc_eval[-1] +
             (labels + 1 == prediction.round()).float().sum().item() +
             (labels - 1 == prediction.round()).float().sum().item()) /
            len(data[0]))

    print("")
    print("    Val Loss:        {}".format(np.average(losses_eval)))
    print("    Val MAE:         {}".format(np.average(losses_eval_mae)))
    print("    Val Acc:         {}".format(np.average(acc_eval)))
    print("    Val 1-off Acc:   {}".format(np.average(acc_eval_one_off)))

    lrSchedular.step(np.average(losses_eval))

    # save model every if current model is better than all previous models
    if len(losses_eval_epochs) == 0 or min(losses_eval_epochs) >= np.average(losses_eval):
        torch.save(net.state_dict(), args.save_dir + "/model_minloss_{}".format(
            '_'.join([key + '_' + str(value) for key, value in vars(args).items()])))
        with open(args.save_dir + "/model_minloss_{}.txt".format('_'.join([key + '_' + str(value) for key, value in vars(args).items()])), "w") as f:
            f.write("Trainig epoch of the saved model: {}".format(e + 1))
    losses_eval_epochs.append(np.average(losses_eval))

    # save model every if current model is better than all previous models (by MAE)
    if len(losses_eval_mae_epochs) == 0 or min(losses_eval_mae_epochs) >= np.average(losses_eval_mae):
        torch.save(net.state_dict(), args.save_dir + "/model_minmae_{}".format(
            '_'.join([key + '_' + str(value) for key, value in vars(args).items()])))
        with open(args.save_dir + "/model_minmae_{}.txt".format('_'.join([key + '_' + str(value) for key, value in vars(args).items()])), "w") as f:
            f.write("Trainig epoch of the saved mode: {}".format(e + 1))
    losses_eval_mae_epochs.append(np.average(losses_eval_mae))

    # test #####################################################################
    counter = 0
    losses_test = []
    losses_test_mae = []
    acc_test = []
    acc_test_one_off = []

    for data in dataloaders['test']:
        counter += 1
        print("\r  Step:     {}/{}".format(counter,
                                           len(dataloaders['val'])), end="")
        features = data[0].cuda()
        labels = data[1].cuda()
        with torch.set_grad_enabled(False):
            prediction = net(features)
            loss = train_criterion(prediction, labels)

        # calculate different types of losses
        losses_test.append(loss.item())
        losses_test_mae.append(
            eval_criterion_mae(prediction, labels.float()).item())
        acc_test.append(
            (labels == prediction.round()).float().sum().item() /
            len(data[0]))
        acc_test_one_off.append(
            (acc_test[-1] +
             (labels + 1 == prediction.round()).float().sum().item() +
             (labels - 1 == prediction.round()).float().sum().item()) /
            len(data[0]))

    print("")
    print("    Test Loss:        {}".format(np.average(losses_test)))
    print("    Test MAE:         {}".format(np.average(losses_test_mae)))
    print("    Test Acc:         {}".format(np.average(acc_test)))
    print("    Test 1-off Acc:   {}".format(np.average(acc_test_one_off)))

    # save the last model
    if e == (starting_epoch + args.epochs - 1) or optimizer.param_groups[0]['lr'] < learningRate * (learningRateFactor ** 5) * 1.01:
        torch.save(net.state_dict(), args.save_dir + "/model_last_{}".format(
            '_'.join([key + '_' + str(value) for key, value in vars(args).items()])))
        break


