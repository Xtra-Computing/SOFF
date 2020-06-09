import numpy as np
import torch
import sys
import copy
import argparse
import pickle

#torch.manual_seed(0)

argv = sys.argv
def printHelp():
    print("Usage: {}".format(argv[0]))

parser = argparse.ArgumentParser(description='train with fedavg and dp')
parser.add_argument('-E', '--epochs', default=30, type=float, help="Number of epochs")
parser.add_argument('-d', '--data-dir', default="preprocessed_data", help="The directory storing preprocessed data")
parser.add_argument('-b', '--batch-size', default = 200, type=int, help="How many record per batch")
parser.add_argument('-a', '--average-every', default = 1, type=int, help="How many epochs per averaging")
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

trainLoaderImdb = DataLoader(trainDatasetImdb, shuffle=True, batch_size = args.batch_size)
validLoaderImdb = DataLoader(validDatasetImdb, shuffle=True, batch_size = args.batch_size)

trainLoaderAmazon = DataLoader(trainDatasetAmazon, shuffle=True, batch_size = args.batch_size)
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

## Define Network #############################################################
print("Defining Network ...")
trainOnGpu = torch.cuda.is_available()
print("    Traning on {}".format("GPU" if trainOnGpu else "CPU"))

from torch import nn

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

totalSize = 0
for _, val in net.named_parameters():
    totalSize = totalSize + val.element_size() * val.nelement();

print("Size of the model:", totalSize)

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
lrSchedulerImdb = torch.optim.lr_scheduler.LambdaLR(optimizerImdb, lambda e: 1/(1+5*e))
lrSchedulerAmazon = torch.optim.lr_scheduler.LambdaLR(optimizerAmazon, lambda e: 1/(1+5*e))

counter = 0
clip = 5

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

        # zero accumnulated grdients
        netImdb.zero_grad()

        # get output, calcualte loss, perform backprop
        outputImdb, h = netImdb(inputsImdb, h)
        lossImdb = criterion(outputImdb.squeeze(), labelsImdb.float())
        lossImdb.backward()

        # `clip_grad_norm` prevents the exploding gradient problem
        nn.utils.clip_grad_norm_(netImdb.parameters(), clip)
        optimizerImdb.step()

        # Update Amazon model ##################################################
        h = tuple([each.data for each in h])

        netAmazon.zero_grad()

        outputAmazon, h = netAmazon(inputsAmazon, h)
        lossAmazon = criterion(outputAmazon.squeeze(), labelsAmazon.float())
        lossAmazon.backward()

        nn.utils.clip_grad_norm_(netAmazon.parameters(), clip)
        optimizerAmazon.step()

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

    if ((e + 1) % args.average_every) == 0:
        params1 = netImdb.named_parameters()
        params2 = netAmazon.named_parameters()
        dict_params2 = dict(params2)
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(beta*param1.data + (1-beta)*dict_params2[name1].data)
        net.load_state_dict(dict_params2)

    # Test every epoch
    testNet()
    lrSchedulerImdb.step()
    lrSchedulerAmazon.step()

## Save Model #################################################################
torch.save(net.state_dict(), savePath)

