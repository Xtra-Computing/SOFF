#!/bin/bash

mkdir -p result_fedavg

python -u train.py --setting=0 | tee -a "./result_fedavg/result_casia" # party 0 (CASIA-HWDB1.1)
python -u train.py --setting=1 | tee -a "./result_fedavg/result_hit"  # party 1 (HIT-OR3C) 
python -u train.py --setting=combined | tee -a "./result_fedavg/result_combined"  # train on combined dataset
python -u train.py --setting=fedavg | tee -a "./result_fedavg/result_fedavg"  # default is fedavg