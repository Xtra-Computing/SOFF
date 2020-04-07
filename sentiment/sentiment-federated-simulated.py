import numpy as np
import torch
import sys
import copy

#torch.manual_seed(0)

argv = sys.argv
def printHelp():
    print("Usage: {}".format(argv[0]))

## Parameters #################################################################
seqLength = 200      # How many features (words) per review to feed to network
trainTestRatio = 0.8 # Train/Test ratio
batchSize = 200      # Batch Size
learningRate = 0.001 # Learning Rate
epochs = 50           # Epoch, 4 achieves decent accuracy and prevents overfitting

printEvery = 10
savePath = "./trained-model"

## Load Data ##################################################################
print("Loading Data ...")
with open('../../data/imdb/imdb_text', 'r') as f:
    reviewsImdb = f.read()
with open('../../data/imdb/imdb_score', 'r') as f:
    labelsImdb = f.read()
with open('../../data/amazon/amazon_text_50000', 'r') as f:
    reviewsAmazon = f.read()
with open('../../data/amazon/amazon_score_50000', 'r') as f:
    labelsAmazon = f.read()

#savePath += "-imdb"
#savePath += "-amazon"

## Data Preprocessing #########################################################
print("Preprocessing data ...")
from string import punctuation

reviewsImdb = reviewsImdb.lower()
reviewsAmazon = reviewsAmazon.lower()

allTextImdb = ''.join([c for c in reviewsImdb if c not in punctuation])
allTextAmazon = ''.join([c for c in reviewsAmazon if c not in punctuation])

reviewsSplitImdb = allTextImdb.split('\n')
reviewsSplitAmazon = allTextAmazon.split('\n')
allTextImdb = ' '.join(reviewsSplitImdb)
allTextAmazon = ' '.join(reviewsSplitAmazon)

words = (allTextImdb + ' ' + allTextAmazon).split()

## Encode Words #############################################################
print("Encoding words ...")
from collections import Counter

# map vocabulary to int, starting from 1
counts = Counter(words)
vocab = sorted(counts, key = counts.get, reverse=True)
vocabToInt = {word: ii for ii, word in enumerate(vocab, 1)}
print('    Unique words: ', len((vocabToInt)))

# map reviews to ints
reviewsAsIntsImdb = []
for review in reviewsSplitImdb:
    reviewsAsIntsImdb.append([vocabToInt[word] for word in review.split()])
reviewsAsIntsAmazon = []
for review in reviewsSplitAmazon:
    reviewsAsIntsAmazon.append([vocabToInt[word] for word in review.split()])

## Encoding Labels ############################################################
print("Encoding labels ...")
labelsSplitImdb = labelsImdb.split('\n')
labelsSplitAmazon = labelsAmazon.split('\n')
encodedLabelsImdb = np.array([1 if label == 'p' else 0 for label in labelsSplitImdb])
encodedLabelsAmazon = np.array([1 if label == 'p' else 0 for label in labelsSplitAmazon])

## Remove Empty Reviews ########################################################
assert(len(encodedLabelsImdb) == len(reviewsAsIntsImdb))
assert(len(encodedLabelsAmazon) == len(reviewsAsIntsAmazon))
encodedLabelsImdb = np.array(list(list(zip(*[(review, label) for review, label in zip(reviewsAsIntsImdb, encodedLabelsImdb) if len(review) != 0]))[1]))
encodedLabelsAmazon = np.array(list(list(zip(*[(review, label) for review, label in zip(reviewsAsIntsAmazon, encodedLabelsAmazon) if len(review) != 0]))[1]))
reviewsAsIntsImdb = [review for review in reviewsAsIntsImdb if len(review) != 0]
reviewsAsIntsAmazon = [review for review in reviewsAsIntsAmazon if len(review) != 0]
assert(len(encodedLabelsImdb) == len(reviewsAsIntsImdb))
assert(len(encodedLabelsAmazon) == len(reviewsAsIntsAmazon))

## Padding Features ###########################################################
print("Padding featuers ...")
def padFeatures(reviewsAsInts, seqLength):
    features = np.zeros((len(reviewsAsInts), seqLength), dtype=int)
    for i, row in enumerate(reviewsAsInts):
        features[i, -min(len(row), seqLength):] = np.array(row)[:seqLength]
    return features

# `featuers` is the final result of all encoded and padded reviews
featuresImdb = padFeatures(reviewsAsIntsImdb, seqLength)
featuresAmazon = padFeatures(reviewsAsIntsAmazon, seqLength)

## Randomize datasets (important) #############################################
seed = np.random.randint(0, 10000)

# shuffle items and labels in the same manner
np.random.seed(seed)
np.random.shuffle(featuresImdb)
np.random.seed(seed)
np.random.shuffle(encodedLabelsImdb)

np.random.seed(seed)
np.random.shuffle(featuresAmazon)
np.random.seed(seed)
np.random.shuffle(encodedLabelsAmazon)

## Splitting datasets #########################################################
print("Splitting Datasets ...")
splitIdxImdb = int(len(featuresImdb) * trainTestRatio)
trainXImdb, remainXImdb = featuresImdb[:splitIdxImdb], featuresImdb[splitIdxImdb:]
trainYImdb, remainYImdb = encodedLabelsImdb[:splitIdxImdb], encodedLabelsImdb[splitIdxImdb:]
testIdxImdb = int(len(remainXImdb) * 0.5)
valXImdb, testXImdb = remainXImdb[:testIdxImdb], remainXImdb[testIdxImdb:]
valYImdb, testYImdb = remainYImdb[:testIdxImdb], remainYImdb[testIdxImdb:]

splitIdxAmazon = int(len(featuresAmazon) * trainTestRatio)
trainXAmazon, remainXAmazon = featuresAmazon[:splitIdxAmazon], featuresAmazon[splitIdxAmazon:]
trainYAmazon, remainYAmazon = encodedLabelsAmazon[:splitIdxAmazon], encodedLabelsAmazon[splitIdxAmazon:]
testIdxAmazon = int(len(remainXAmazon) * 0.5)
valXAmazon, testXAmazon = remainXAmazon[:testIdxAmazon], remainXAmazon[testIdxAmazon:]
valYAmazon, testYAmazon = remainYAmazon[:testIdxAmazon], remainYAmazon[testIdxAmazon:]

print("    Features Shapes:")
print("        Imdb:       ")
print("            Train:      {}/{}".format(trainXImdb.shape, trainYImdb.shape))
print("            Validation: {}/{}".format(valXImdb.shape, valYImdb.shape))
print("            Test:       {}/{}".format(testXImdb.shape, testYImdb.shape))
print("        Amazon:     ")
print("            Train:      {}/{}".format(trainXAmazon.shape, trainYAmazon.shape))
print("            Validation: {}/{}".format(valXAmazon.shape, valYAmazon.shape))
print("            Test:       {}/{}".format(testXAmazon.shape, testYAmazon.shape))

################################################# Dataset PreProcessing Done ##
## Starting Federeated Data Partition and Training ############################

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

trainLoaderImdb = DataLoader(trainDatasetImdb, shuffle=True, batch_size = batchSize)
validLoaderImdb = DataLoader(validDatasetImdb, shuffle=True, batch_size = batchSize)

trainLoaderAmazon = DataLoader(trainDatasetAmazon, shuffle=True, batch_size = batchSize)
validLoaderAmazon = DataLoader(validDatasetAmazon, shuffle=True, batch_size = batchSize)

testLoaders = { name: DataLoader(dataset,  shuffle=True, batch_size = batchSize) for name, dataset in testDatasets.items() }

#trainDatasets = [sy.BaseDataset(torch.from_numpy(trainXImdb).send(bob), torch.from_numpy(trainYImdb).send(bob))]
#federatedTrainLoader = sy.FederatedDataLoader(sy.FederatedDataset(trainDatasets), shuffle=True, batch_size = batchSize)
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

vocabSize = len(vocabToInt) + 1
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
        h = net.init_hidden(batchSize)

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

for e in range(epochs):
    h = net.init_hidden(batchSize)

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

            validH = net.init_hidden(batchSize)
            validLosses = []
            for inputs, labels in validLoaderImdb:
                validH = tuple([each.data for each in validH])
                if(trainOnGpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, validH = net(inputs, validH)
                validLoss = criterion(output.squeeze(), labels.float())
                validLosses.append(validLoss.item())
            print("    Valid using imdb data: ")
            print("        Epoch:    {}/{}".format(e+1, epochs))
            print("        Step:     {}".format(counter))
            print("        Loss:     {:.6f}".format(lossImdb.item()))
            print("        Val Loss: {:.6f}".format(np.mean(validLosses)))

            validH = net.init_hidden(batchSize)
            validLosses = []
            for inputs, labels in validLoaderAmazon:
                validH = tuple([each.data for each in validH])
                if(trainOnGpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, validH = net(inputs, validH)
                validLoss = criterion(output.squeeze(), labels.float())
                validLosses.append(validLoss.item())
            print("    Valid using amazon data: ")
            print("        Epoch:    {}/{}".format(e+1, epochs))
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
    lrSchedulerImdb.step()
    lrSchedulerAmazon.step()

## Save Model #################################################################
torch.save(net.state_dict(), savePath)

