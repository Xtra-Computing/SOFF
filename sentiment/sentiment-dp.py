import numpy as np
import torch
import sys
import argparse
import pickle
from scipy.optimize import newton,bisect
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

#torch.manual_seed(0)

parser = argparse.ArgumentParser(description='train with fedavg and dp')
parser.add_argument('-D', '--dataset', default='imdb', type=str, help="imdb | amazon | both")
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

if args.dataset == "imdb":
    trainX = trainXImdb
    trainY = trainYImdb
    trainDataset = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY))
    validDataset = TensorDataset(torch.from_numpy(valXImdb),   torch.from_numpy(valYImdb))
elif args.dataset == "amazon":
    trainX = trainXAmazon
    trainY = trainYAmazon
    trainDataset = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY))
    validDataset = TensorDataset(torch.from_numpy(valXAmazon),   torch.from_numpy(valYAmazon))
elif args.dataset == "both":
    trainX = np.concatenate([trainXImdb, trainXAmazon])
    trainY = np.concatenate([trainYImdb, trainYAmazon])
    trainDataset = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY))
    validDataset = TensorDataset(
        torch.from_numpy(np.concatenate([valXImdb, valXAmazon])),
        torch.from_numpy(np.concatenate([valYImdb, valYAmazon])))
else:
    exit(1)

testDatasets = {
    'imdb':     TensorDataset(torch.from_numpy(testXImdb),  torch.from_numpy(testYImdb)),
    'amazon':   TensorDataset(torch.from_numpy(testXAmazon),  torch.from_numpy(testYAmazon)),
    'both':     TensorDataset(
        torch.from_numpy(np.concatenate([testXImdb, testXAmazon])),
        torch.from_numpy(np.concatenate([testYImdb, testYAmazon])))
}

sampler = torch.utils.data.RandomSampler(trainDataset, replacement=True)
trainLoader = DataLoader(trainDataset, shuffle=False, sampler=sampler, batch_size = args.batch_size)
validLoader = DataLoader(validDataset, shuffle=True, batch_size = args.batch_size)
testLoaders = { name: DataLoader(dataset,  shuffle=True, batch_size = args.batch_size) for name, dataset in testDatasets.items() }

#trainDatasets = [sy.BaseDataset(torch.from_numpy(trainXImdb).send(bob), torch.from_numpy(trainYImdb).send(bob))]
#federatedTrainLoader = sy.FederatedDataLoader(sy.FederatedDataset(trainDatasets), shuffle=True, batch_size = args.batch_size)
#print("    Workers: ", federatedTrainDataset.workers)

# print a sample
dataIter = iter(trainLoader)
sampleX, sampleY = dataIter.next()
print('    Sample input size:  ', sampleX.size())  # batch_size, seq_length
print('    Sample label size:  ', sampleY.size())  # batch_size

## DP Config ###################################################################

clip = 0.1

lotSize = args.batches_per_lot * args.batch_size # L
delta = min(10**(-5), 1/trainX.shape[0])

lotsPerEpoch = trainX.shape[0]/ lotSize
q = lotSize / trainX.shape[0]
T = args.epochs * lotsPerEpoch

def compute_dp_sgd_wrapper(_sigma):
    return compute_dp_sgd_privacy.compute_dp_sgd_privacy(
            n=trainX.shape[0],
            batch_size=lotSize,
            noise_multiplier=_sigma,
            epochs=args.epochs,
            delta=delta)[0] - args.epsilon
# sigma = newton(compute_dp_sgd_wrapper, 0.66/np.sqrt(args.epsilon))
sigma = bisect(compute_dp_sgd_wrapper, 0.01, 10000)

print('BpL={}, q={}, T={}, σ₁=σ₂={}'.format(args.batches_per_lot, q, T, sigma))
print('actual epslion = {}'.format(
    compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=trainX.shape[0], batch_size=lotSize,
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

class SentimentRNN(nn.Module):
    def __init__(self, vocabSize, outputSize, embeddingDim, hiddenDim, nLayers, dropProb = 0.2):
        super(SentimentRNN, self).__init__()
        self.outputSize = outputSize
        self.nLayers = nLayers
        self.hiddenDim = hiddenDim

        self.embedding = nn.Embedding(vocabSize, embeddingDim)
        self.lstm = nn.LSTM(embeddingDim, hiddenDim, nLayers, dropout = 0, batch_first = True)
        self.dropout = nn.Dropout(dropProb)

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
# optimizer = torch.optim.SGD(net.parameters(), lr = learningRate)
optimizer = torch.optim.Adam(net.parameters(), lr = learningRate)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1/(1+5*e))
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.2)

counter = 0

if trainOnGpu:
    net.cuda()

net.train()
for e in range(args.epochs):
    h = net.init_hidden(args.batch_size)

    for inputs, labels in trainLoader:
        counter += 1
        if trainOnGpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # get output, calcualte loss, perform backprop
        output, h = net(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        # `clip_grad_norm` prevents the exploding gradient problem
        if counter % args.batches_per_lot == 0:
            for _, param in net.named_parameters():
                param.grad /= args.batches_per_lot
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            if not args.no_noise:
                net.add_gaussian_noise()
            optimizer.step()
            net.zero_grad()

        if counter % printEvery == 0:
            net.eval()

            validH = net.init_hidden(args.batch_size)
            validLosses = []
            net.eval()
            for inputs, labels in validLoader:
                validH = tuple([each.data for each in validH])
                if(trainOnGpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, validH = net(inputs, validH)
                validLoss = criterion(output.squeeze(), labels.float())
                validLosses.append(validLoss.item())

            net.train()
            print("    Epoch:    {}/{}".format(e+1, args.epochs))
            print("    Step:     {}".format(counter))
            print("    Loss:     {:.6f}".format(loss.item()))
            print("    Val Loss: {:.6f}".format(np.mean(validLosses)))
            print("")

    # Test every epoch
    testNet()

    #lr_scheduler.step()

## Save Model #################################################################
torch.save(net.state_dict(), savePath)

#net.load_state_dict(torch.load(savePath))
#net.eval()

