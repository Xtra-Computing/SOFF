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


available_datasets = ['aligned', 'union']

# fmt: off
parser = argparse.ArgumentParser(description='train a year prediction model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-E', '--epochs', default=1000, type=int, help="Number of epochs")
parser.add_argument('-d', '--data-dir', default="preprocessed_data", help="The directory storing preprocessed data")
parser.add_argument('-b', '--batch-size', default=256, type=int, help="How many record per batch")
parser.add_argument('-a', '--average-every', default=1, type=int, help="How many epochs per averaging")
parser.add_argument('-s', '--save-dir', default="saves", type=str, help="Model save dir")
parser.add_argument('-p', '--patience', default=30, type=int, help="LRSchedular patience")
parser.add_argument('-o', '--optimizer', default='sgd', type=str, help="sgd|adam")
# splitnn is enabled by default
# parser.add_argument('-sn', '--splitnn', action='store_true', help="simulate splitNN, when using aligned or union dataset")
parser.add_argument('-lr', '--learning-rate', default=1e-7, type=float, help="Starting learning rate")
parser.add_argument('-ds', '--dataset', default='fma', type=str, help='|'.join(available_datasets))
args = parser.parse_args()
# fmt: on

if not args.dataset in available_datasets:
    raise Exception("Unrecognized dataset")

print(args)

## Initialize data loader #######################################################
print("Loading data...")

if args.dataset == 'aligned':
    dataset = AlignedDataset(args.data_dir, join_data=False)
elif args.dataset == 'union':
    dataset = UnionDataset(args.data_dir, join_data=False)
else:
    raise Exception("No such dataset")

train_size = int(len(dataset) * 0.9)
val_size = int((len(dataset) - train_size) * 0.5)
test_size = len(dataset) - train_size - val_size

# we need to fix the splitting random seed so that the three aligned datasets use the same indices as test set
torch.manual_seed(42)  # use this for torch < 1.6 compatibility
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


class ClientNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

        self.l1 = nn.Linear(input_size, hidden_size_1, bias=True)
        self.l2 = nn.Linear(hidden_size_1, hidden_size_2, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        return x


class ServerNet(nn.Module):
    def __init__(self, input_sizes):
        super().__init__()
        self.output = nn.Linear(np.sum(input_sizes), 1)

    def forward(self, inputs):
        x = torch.cat(tuple(inputs), dim=1)
        x = self.output(x)
        return x


net_clients = [ClientNet(input_size, input_size * 2, input_size // 2).cuda()
               for input_size in dataset.input_sizes()]
net_server = ServerNet([input_size // 2 for input_size in dataset.input_sizes()]).cuda()

# if args.load_model:
#    print("Loading saved model...")
#    net.load_state_dict(torch.load(args.load_model))

learningRate = args.learning_rate
learningRateFactor = 0.1

if args.optimizer == 'sgd':
    optimizer_server = torch.optim.SGD(
        net_server.parameters(), lr=learningRate, momentum=0.9)
    optimizer_clients = [torch.optim.SGD(
        net.parameters(), lr=learningRate, momentum=0.9) for net in net_clients]
elif args.optimizer == 'adam':
    optimizer_server = torch.optim.Adam(
        net_server.parameters(), lr=learningRate)
    optimizer_clients = [torch.optim.Adam(
        net.parameters()) for net in net_clients]
else:
    raise Excecption("Unkown optimizer: {}".format(args.optimizer))

lr_schedular_server = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_server, mode='min', factor=learningRateFactor, patience=args.patience, verbose=True)

lr_schedular_clients = [torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=learningRateFactor, patience=args.patience, verbose=True) for optimizer in optimizer_clients]


train_criterion = nn.MSELoss()
eval_criterion_mae = nn.L1Loss()

train_criterion.cuda()
eval_criterion_mae.cuda()

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
    net_server.train()
    for net in net_clients:
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

        featuress = [d.cuda() for d in data[:-1]]
        labels = data[-1].cuda()

        with torch.set_grad_enabled(True):
            optimizer_server.zero_grad()
            for optimizer in optimizer_clients:
                optimizer.zero_grad()
            # client prediction
            prediction_clients = [net(features)
                                  for net, features in zip(net_clients, featuress)]

            # send tensors to server, server continue prediction
            # TODO: the copy causes buffer overflow
            prediction_clients_copy = [
                torch.zeros_like(prediction_client) for prediction_client in prediction_clients]
            for prediction_copy, prediction_client in zip(prediction_clients_copy, prediction_clients):
                prediction_copy.data.copy_(prediction_client)
                prediction_copy.requires_grad_(True)
            prediction = net_server(prediction_clients_copy)
            loss = train_criterion(prediction, labels)
            loss.backward()
            optimizer_server.step()

            # server send back the split layer (prediction_clients_copy) to client, client continue back prop
            for prediction_client, prediction_copy in zip(prediction_clients, prediction_clients_copy):
                prediction_client.backward(prediction_copy.grad)
            for optimizer in optimizer_clients:
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
    net_server.eval()
    for net in net_clients:
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
        featuress = [d.cuda() for d in data[:-1]]
        labels = data[-1].cuda()
        with torch.set_grad_enabled(False):
            prediction_clients = [net(features)
                                  for net, features in zip(net_clients, featuress)]
            prediction = net_server(prediction_clients)
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

    lr_schedular_server.step(np.average(losses_eval))
    map(lambda schedular: schedular.step(losses_eval), lr_schedular_clients)

    # save model every if current model is better than all previous models
    # if len(losses_eval_epochs) == 0 or min(losses_eval_epochs) >= np.average(losses_eval):
    #     torch.save(net.state_dict(), args.save_dir + "/model_minloss_{}".format(
    #         '_'.join([key + '_' + str(value) for key, value in vars(args).items()])))
    #     with open(args.save_dir + "/model_minloss_{}.txt".format('_'.join([key + '_' + str(value) for key, value in vars(args).items()])), "w") as f:
    #         f.write("Trainig epoch of the saved model: {}".format(e + 1))
    # losses_eval_epochs.append(np.average(losses_eval))

    # # save model every if current model is better than all previous models (by MAE)
    # if len(losses_eval_mae_epochs) == 0 or min(losses_eval_mae_epochs) >= np.average(losses_eval_mae):
    #     torch.save(net.state_dict(), args.save_dir + "/model_minmae_{}".format(
    #         '_'.join([key + '_' + str(value) for key, value in vars(args).items()])))
    #     with open(args.save_dir + "/model_minmae_{}.txt".format('_'.join([key + '_' + str(value) for key, value in vars(args).items()])), "w") as f:
    #         f.write("Trainig epoch of the saved mode: {}".format(e + 1))
    # losses_eval_mae_epochs.append(np.average(losses_eval_mae))

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
        featuress = [d.cuda() for d in data[:-1]]
        labels = data[-1].cuda()
        with torch.set_grad_enabled(False):
            prediction_clients = [net(features)
                                  for net, features in zip(net_clients, featuress)]
            prediction = net_server(prediction_clients)
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

    if e == (starting_epoch + args.epochs - 1) or optimizer_server.param_groups[0]['lr'] < learningRate * (learningRateFactor ** 5) * 1.01:
        break
    # save model
    #     torch.save(net.state_dict(), args.save_dir + "/model_last_{}".format(
    #         '_'.join([key + '_' + str(value) for key, value in vars(args).items()])))
    #     break
