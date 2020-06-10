#!/bin/bash

# Improvement of federation
python train.py --setting=0  # party 0 (CASIA-HWDB1.1)
python train.py --setting=1  # party 1 (HIT-OR3C)
python train.py --setting=combined  # train on combined dataset
python train.py --setting=fedavg  # default is fedavg