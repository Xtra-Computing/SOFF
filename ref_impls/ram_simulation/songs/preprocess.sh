#!/bin/bash

mkdir -p data

curl -L http://millionsongdataset.com/sites/default/files/AdditionalFiles/msd_summary_file.h5 -o data/msd_summary_file.h5
curl -L http://millionsongdataset.com/sites/default/files/AdditionalFiles/tracks_per_year.txt -o data/tracks_per_year.txt

curl -L https://os.unil.cloud.switch.ch/fma/fma_metadata.zip -o data/fma_metadata.zip
unzip data/fma_metadata.zip -d data/

python -u preprocess.py
