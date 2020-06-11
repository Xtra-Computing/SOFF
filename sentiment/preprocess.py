import numpy as np
import torch
import sys
import copy
import argparse
import os
import pickle
import errno

parser = argparse.ArgumentParser(description='Preprocess data')
parser.add_argument('-s','--save-dir', default="preprocessed_data")
parser.add_argument('-l','--seq-length', default=200, type=int, help="data longer than this many characters are clipped to this value")
parser.add_argument('-t','--train-test-ratio', default=0.8, type=float)
parser.add_argument('-n','--negative-encoding', action='store_true', help="set negative to -1 and positive to 1. If not specified, set negative to 0 and positive to 1")
args = parser.parse_args()

seqLength = args.seq_length               # How many features (words) per review to feed to network
trainTestRatio = args.train_test_ratio    # Train/Test ratio

## Load Data ##################################################################
print("Loading Data ...")
with open('./data/imdb/imdb_text', 'r') as f:
    reviewsImdb = f.read()
with open('./data/imdb/imdb_score', 'r') as f:
    labelsImdb = f.read()
with open('./data/amazon/amazon_text_50000', 'r') as f:
    reviewsAmazon = f.read()
with open('./data/amazon/amazon_score_50000', 'r') as f:
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
encodedLabelsImdb = np.array([1 if label == 'p' else (-1 if args.negative_encoding else 0) for label in labelsSplitImdb])
encodedLabelsAmazon = np.array([1 if label == 'p' else (-1 if args.negative_encoding else 0) for label in labelsSplitAmazon])

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

try:
    os.mkdir(args.save_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

np.save(args.save_dir + '/trainXImdb', trainXImdb, allow_pickle=False)
np.save(args.save_dir + '/trainYImdb', trainYImdb, allow_pickle=False)
np.save(args.save_dir + '/valXImdb', valXImdb, allow_pickle=False)
np.save(args.save_dir + '/valYImdb', valYImdb, allow_pickle=False)
np.save(args.save_dir + '/testXImdb', testXImdb, allow_pickle=False)
np.save(args.save_dir + '/testYImdb', testYImdb, allow_pickle=False)

np.save(args.save_dir + '/trainXAmazon', trainXAmazon, allow_pickle=False)
np.save(args.save_dir + '/trainYAmazon', trainYAmazon, allow_pickle=False)
np.save(args.save_dir + '/valXAmazon', valXAmazon, allow_pickle=False)
np.save(args.save_dir + '/valYAmazon', valYAmazon, allow_pickle=False)
np.save(args.save_dir + '/testXAmazon', testXAmazon, allow_pickle=False)
np.save(args.save_dir + '/testYAmazon', testYAmazon, allow_pickle=False)

info = {
        'vocabToIntLength': len(vocabToInt),
        }

with open(args.save_dir + '/info.pkl', 'wb') as f:
    pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)

