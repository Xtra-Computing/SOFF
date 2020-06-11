#!/bin/bash

set -e  # exit on error

mkdir -p data

pushd data  # cd data

# skip download if the file exists
wget -nc http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart1.zip
wget -nc http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart2.zip
wget -nc http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1Test.zip
wget -nc http://www.iapr-tc11.org/dataset/OR3C_DAS2010/v1.1/OR3C/offline/character.rar
wget -nc https://www.rarlab.com/rar/rarlinux-x64-5.9.0.tar.gz

mkdir -p character/trn
mkdir -p character/tst

unzip Gnt1.1TrainPart1.zip -d Gnt1.1Train
unzip Gnt1.1TrainPart2.zip -d Gnt1.1Train
unzip Gnt1.1Test.zip -d Gnt1.1Test

tar -xzvf rarlinux-x64-5.9.0.tar.gz
rar/rar x character.rar

for ((i=1; i<99; ++i))  
do  
    mv character/${i}_images character/trn/
done

for ((i=99; i<123; ++i))  
do  
    mv character/${i}_images character/tst/
done

python preprocess.py

popd