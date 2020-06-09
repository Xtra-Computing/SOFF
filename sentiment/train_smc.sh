#!/bin/bash

mkdir -p result_smc

python -u ./sentiment-federated-smc.py -s | tee -a "./result_smc/epochs_30_batchsize_200_smc"
python -u ./sentiment-federated-smc.py    | tee -a "./result_smc/epochs_30_batchsize_200_nosmc"

