#!/bin/bash

# Improvement of federation
python train.py --setting=0 2>&1 | tee improve-set0.txt # Caucasian
python train.py --setting=1 2>&1 | tee improve-set1.txt # African
python train.py --setting=combined 2>&1 | tee improve-setcom.txt
python train.py --setting=fedavg 2>&1 | tee improve-setfed.txt