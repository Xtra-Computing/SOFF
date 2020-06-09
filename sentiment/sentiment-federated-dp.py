import numpy as np
import torch
import sys
import copy
import argparse
import pickle
from scipy.optimize import newton,bisect
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

#torch.manual_seed(0)

parser = argparse.ArgumentParser(description='train with fedavg and dp')
parser.add_argument('-E', '--epochs', default=10, type=int, help="Number of epochs")
parser.add_argument('-d', '--data-dir', default="preprocessed_data", help="The directory storing preprocessed data")
parser.add_argument('-e', '--epsilon', default=1.0, type=float, help="Privacy Budget")
parser.add_argument('-b', '--batch-size', default = 40, type=int, help="How many record per batch")
parser.add_argument('-l', '--batches-per-lot', default = 10, type=int, help="How many batches per lot")
parser.add_argument('-n', '--no-noise', action='store_true')
args = parser.parse_args()

print(args)

## Parameters #################################################################
learningRate = 0.001 # Learning Rate
#learningRate = 1 # SGD test

printEvery = 10
savePath = "./trained-model"

## Load Preprocessed Data #####################################################

trainXImdb = np.load(args.data_dir + '/trainXImdb.npy')
trainYImdb = np.load(args.data_dir + '/trainYImdb.npy')
valXImdb   = np.load(args.data_dir + '/valXImdb.npy')
valYImdb   = np.load(args.data_dir + '/valYImdb.npy')
testXImdb  = np.load(args.data_dir + '/testXImdb.npy')
testYImdb  = np.load(args.data_dir + '/testYImdb.npy')

trainXAmazon =np.load(args.data_dir + '/trainXAmazon.npy')
trainYAmazon =np.load(args.data_dir + '/trainYAmazon.npy')
valXAmazon   =np.load(args.data_dir + '/valXAmazon.npy')
valYAmazon   =np.load(args.data_dir + '/valYAmazon.npy')
testXAmazon  =np.load(args.data_dir + '/testXAmazon.npy')
testYAmazon  =np.load(args.data_dir + '/testYAmazon.npy')

## Define DataLoaders and Virtual Workers #####################################
print("Building dataLoaders ...")
from torch.utils.data import TensorDataset, DataLoader
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#import syft as sy
#hook = sy.TorchHook(torch)
#alice = sy.VirtualWorker(hook, id="alice")
#bob = sy.VirtualWorker(hook, id="bob")

trainDatasetImdb = TensorDataset(torch.from_numpy(trainXImdb), torch.from_numpy(trainYImdb))
validDatasetImdb = TensorDataset(torch.from_numpy(valXImdb),   torch.from_numpy(valYImdb))

trainDatasetAmazon = TensorDataset(torch.from_numpy(trainXAmazon), torch.from_numpy(trainYAmazon))
validDatasetAmazon = TensorDataset(torch.from_numpy(valXAmazon),   torch.from_numpy(valYAmazon))

testDatasets = {
    'imdb':     TensorDataset(torch.from_numpy(testXImdb),  torch.from_numpy(testYImdb)),
    'amazon':   TensorDataset(torch.from_numpy(testXAmazon),  torch.from_numpy(testYAmazon)),
    'both':     TensorDataset(
        torch.from_numpy(np.concatenate([testXImdb, testXAmazon])),
        torch.from_numpy(np.concatenate([testYImdb, testYAmazon])))
}

samplerImdb = torch.utils.data.RandomSampler(trainDatasetImdb, replacement=True)
trainLoaderImdb = DataLoader(trainDatasetImdb, shuffle=False, sampler = samplerImdb, batch_size = args.batch_size)
validLoaderImdb = DataLoader(validDatasetImdb, shuffle=True, batch_size = args.batch_size)

samplerAmazon = torch.utils.data.RandomSampler(trainDatasetAmazon, replacement=True)
trainLoaderAmazon = DataLoader(trainDatasetAmazon, shuffle=False, sampler = samplerAmazon, batch_size = args.batch_size)
validLoaderAmazon = DataLoader(validDatasetAmazon, shuffle=True, batch_size = args.batch_size)

testLoaders = { name: DataLoader(dataset,  shuffle=True, batch_size = args.batch_size) for name, dataset in testDatasets.items() }

#trainDatasets = [sy.BaseDataset(torch.from_numpy(trainXImdb).send(bob), torch.from_numpy(trainYImdb).send(bob))]
#federatedTrainLoader = sy.FederatedDataLoader(sy.FederatedDataset(trainDatasets), shuffle=True, batch_size = args.batch_size)
#print("    Workers: ", federatedTrainDataset.workers)

# print a sample
dataIterImdb = iter(trainLoaderImdb)
sampleXImdb, sampleYImdb = dataIterImdb.next()
print('    Imdb:  ')  # batch_size, seq_length
print('        Sample input size:  ', sampleXImdb.size())  # batch_size, seq_length
print('        Sample label size:  ', sampleYImdb.size())  # batch_size

dataIterAmazon = iter(trainLoaderAmazon)
sampleXAmazon, sampleYAmazon = dataIterAmazon.next()
print('    Amazon:  ')  # batch_size, seq_length
print('        Sample input size:  ', sampleXAmazon.size())  # batch_size, seq_length
print('        Sample label size:  ', sampleYAmazon.size())  # batch_size

## DP Config ###################################################################

clip = 0.1

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

assert(trainXImdb.shape[0] == trainXAmazon.shape[0]) # otherwise we need two qs and Ts

lotSize = args.batches_per_lot * args.batch_size # L
delta = min(10**(-5), 1/trainXImdb.shape[0])

lotsPerEpoch = trainXImdb.shape[0]/ lotSize
q = lotSize / trainXImdb.shape[0]
T = args.epochs * lotsPerEpoch

def compute_dp_sgd_wrapper(_sigma):
    return compute_dp_sgd_privacy.compute_dp_sgd_privacy(
            n=trainXImdb.shape[0],
            batch_size=lotSize,
            noise_multiplier=_sigma,
            epochs=args.epochs,
            delta=delta)[0] - args.epsilon
# sigma = newton(compute_dp_sgd_wrapper, 0.66/np.sqrt(args.epsilon))
sigma = bisect(compute_dp_sgd_wrapper, 0.01, 10000)

print('BpL={}, q={}, T={}, σ₁=σ₂={}'.format(args.batches_per_lot, q, T, sigma))
print('actual epslion = {}'.format(
    compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=trainXImdb.shape[0], batch_size=lotSize,
        noise_multiplier=sigma,
        epochs=args.epochs, delta=delta)))

# using global variable
_lastNoiseShape = None
_noiseToAdd = None
def gaussian_noise(grads):
    global _lastNoiseShape
    global _noiseToAdd
    if grads.shape != _lastNoiseShape:
        _lastNoiseShape = grads.shape
        _noiseToAdd = torch.zeros(grads.shape).cuda()
    _noiseToAdd.data.normal_(0.0, std = sigma * clip)
    return _noiseToAdd

## Define Network #############################################################
print("Defining Network ...")
trainOnGpu = torch.cuda.is_available()
print("    Traning on {}".format("GPU" if trainOnGpu else "CPU"))

from torch import nn
from torch.autograd import Variable

class SentimentRNN(nn.Module):
    def __init__(self, vocabSize, outputSize, embeddingDim, hiddenDim, nLayers, dropProb = 0.5):
        super(SentimentRNN, self).__init__()
        self.outputSize = outputSize
        self.nLayers = nLayers
        self.hiddenDim = hiddenDim

        self.embedding = nn.Embedding(vocabSize, embeddingDim)
        self.lstm = nn.LSTM(embeddingDim, hiddenDim, nLayers, dropout = dropProb, batch_first = True)
        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hiddenDim, outputSize)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batchSize = x.size(0)

        embeds = self.embedding(x)
        lstmOut, hidden = self.lstm(embeds, hidden)

        lstmOut = lstmOut.contiguous().view(-1, self.hiddenDim)

        out = self.dropout(lstmOut)
        out = self.fc(out)

        sigmoidOut = self.sig(out)
        sigmoidOut = sigmoidOut.view(batchSize, -1)
        sigmoidOut = sigmoidOut[:, -1]

        return sigmoidOut, hidden

    def init_hidden(self, batchSize):
        weight = next(self.parameters()).data

        if(trainOnGpu):
            hidden = (
                weight.new(self.nLayers, batchSize, self.hiddenDim).zero_().cuda(),
                weight.new(self.nLayers, batchSize, self.hiddenDim).zero_().cuda())
        else:
            hidden = (
                weight.new(self.nLayers, batchSize, self.hiddenDim).zero_(),
                weight.new(self.nLayers, batchSize, self.hiddenDim).zero_())

        return hidden


    def clip_grad_to_bound(self):
        for key, param in self.named_parameters():
            param.grad /= n_batch
            gradient_clip(param)

    def add_gaussian_noise(self):
        for key, param in self.named_parameters():
            noise = 1/lotSize * gaussian_noise(param.grad)
            param.grad += noise


## Instantiate Network ########################################################
print("Instantiating Network ...")

with open(args.data_dir + '/info.pkl', 'rb') as f:
    info = pickle.load(f)

vocabSize = info['vocabToIntLength'] + 1
outputSize = 1
embeddingDim = 512
hiddenDim = 256
nLayers = 2

net = SentimentRNN(vocabSize, outputSize, embeddingDim, hiddenDim, nLayers)
netImdb   = SentimentRNN(vocabSize, outputSize, embeddingDim, hiddenDim, nLayers)
netAmazon = SentimentRNN(vocabSize, outputSize, embeddingDim, hiddenDim, nLayers)


## Define Testing Function ####################################################
def testNet():
    net.eval()
    for name, testLoader in testLoaders.items():
        testLosses = []
        numCorrect = 0
        h = net.init_hidden(args.batch_size)

        for inputs, labels in testLoader:
            h = tuple([each.data for each in h])
            if(trainOnGpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            output, h = net(inputs, h)
            testlLoss = criterion(output.squeeze(), labels.float())
            testLosses.append(testlLoss.item())

            pred = torch.round(output.squeeze())

            correctTensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correctTensor.numpy()) if not trainOnGpu else np.squeeze(correctTensor.cpu().numpy())
            numCorrect += np.sum(correct)

        testAcc = numCorrect / len(testLoader.dataset)
        print("Test Accuracy ({}): {:.3f}".format(name, testAcc))
    net.train()

## Train Network ##############################################################
print("Training ...")

criterion = nn.BCELoss()

optimizerImdb = torch.optim.Adam(netImdb.parameters(), lr = learningRate)
optimizerAmazon = torch.optim.Adam(netAmazon.parameters(), lr = learningRate)
# lrSchedulerImdb = torch.optim.lr_scheduler.LambdaLR(optimizerImdb, lambda e: 1/(1+5*e))
# lrSchedulerAmazon = torch.optim.lr_scheduler.LambdaLR(optimizerAmazon, lambda e: 1/(1+5*e))

counter = 0

if trainOnGpu:
    net.cuda()
    netImdb.cuda()
    netAmazon.cuda()
    print("    Traninig on cuda ...")

net.train()
netImdb.train()
netAmazon.train()

for e in range(args.epochs):
    h = net.init_hidden(args.batch_size)

    # Update client model ##################################################
    netImdb.load_state_dict(copy.deepcopy(net.state_dict()))
    netAmazon.load_state_dict(copy.deepcopy(net.state_dict()))

    for (inputsImdb, labelsImdb), (inputsAmazon, labelsAmazon) in zip(trainLoaderImdb, trainLoaderAmazon):
        counter += 1
        if trainOnGpu:
            inputsImdb, labelsImdb = inputsImdb.cuda(), labelsImdb.cuda()
            inputsAmazon, labelsAmazon = inputsAmazon.cuda(), labelsAmazon.cuda()

        # Update Imdb model ####################################################
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # get output, calcualte loss, perform backprop
        outputImdb, h = netImdb(inputsImdb, h)
        lossImdb = criterion(outputImdb.squeeze(), labelsImdb.float())
        lossImdb.backward()

        # `clip_grad_norm` prevents the exploding gradient problem
        if counter % args.batches_per_lot == 0:
#            if(counter / args.batches_per_lot == 5):
#                for _, val in netImdb.named_parameters():
#                    print("max:",torch.max(val.grad))
#                    print("min:",torch.min(val.grad))

            for _, param in netImdb.named_parameters():
                param.grad /= args.batches_per_lot
            nn.utils.clip_grad_norm_(netImdb.parameters(), clip)
            if not args.no_noise:
                netImdb.add_gaussian_noise()
            optimizerImdb.step()
            netImdb.zero_grad()

        # Update Amazon model ##################################################
        h = tuple([each.data for each in h])

        outputAmazon, h = netAmazon(inputsAmazon, h)
        lossAmazon = criterion(outputAmazon.squeeze(), labelsAmazon.float())
        lossAmazon.backward()

        if counter % args.batches_per_lot == 0:
#            if(counter / args.batches_per_lot == 5):
#                for _, val in netAmazon.named_parameters():
#                    print("max:",torch.max(val.grad))
#                    print("min:",torch.min(val.grad))
#                exit(1);

            for _, param in netAmazon.named_parameters():
                param.grad /= args.batches_per_lot
            nn.utils.clip_grad_norm_(netAmazon.parameters(), clip)
            if not args.no_noise:
                netAmazon.add_gaussian_noise()
            optimizerAmazon.step()
            netAmazon.zero_grad()

        if counter % printEvery == 0:
            net.eval()

            validH = net.init_hidden(args.batch_size)
            validLosses = []
            for inputs, labels in validLoaderImdb:
                validH = tuple([each.data for each in validH])
                if(trainOnGpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, validH = net(inputs, validH)
                validLoss = criterion(output.squeeze(), labels.float())
                validLosses.append(validLoss.item())
            print("    Valid using imdb data: ")
            print("        Epoch:    {}/{}".format(e+1, args.epochs))
            print("        Step:     {}".format(counter))
            print("        Loss:     {:.6f}".format(lossImdb.item()))
            print("        Val Loss: {:.6f}".format(np.mean(validLosses)))

            validH = net.init_hidden(args.batch_size)
            validLosses = []
            for inputs, labels in validLoaderAmazon:
                validH = tuple([each.data for each in validH])
                if(trainOnGpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, validH = net(inputs, validH)
                validLoss = criterion(output.squeeze(), labels.float())
                validLosses.append(validLoss.item())
            print("    Valid using amazon data: ")
            print("        Epoch:    {}/{}".format(e+1, args.epochs))
            print("        Step:     {}".format(counter))
            print("        Loss:     {:.6f}".format(lossAmazon.item()))
            print("        Val Loss: {:.6f}".format(np.mean(validLosses)))

            print("")
            net.train()

    # Average two model (to simluate fedavg) ###############################
    beta = 0.5
    params1 = netImdb.named_parameters()
    params2 = netAmazon.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(beta*param1.data + (1-beta)*dict_params2[name1].data)
    net.load_state_dict(dict_params2)

    # Test every epoch
    testNet()
    # lrSchedulerImdb.step()
    # lrSchedulerAmazon.step()

## Save Model #################################################################
torch.save(net.state_dict(), savePath)

