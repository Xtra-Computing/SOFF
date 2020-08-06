#!/bin/bash

set -e  # exit on error

mkdir -p data

pushd data  # cd data

# skip download if the file exists
wget -nc http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz
wget -nc http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz

tar zxvf EnglishFnt.tgz
tar zxvf EnglishHnd.tgz

python ../preprocess.py

popd