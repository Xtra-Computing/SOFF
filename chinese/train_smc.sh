#!/bin/bash

mkdir -p result_smc

python -u ./train.py -E 20 --setting fedavg --spdz | tee -a "./result_smc/epochs_20_batchsize_128_smc"
python -u ./train.py -E 20 --setting fedavg        | tee -a "./result_smc/epochs_20_batchsize_128_nosmc"
