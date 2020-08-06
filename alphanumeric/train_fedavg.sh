#!/bin/bash

# Improvement of federation
python train.py --setting=0 2>&1 | tee improve-set0.txt # fnt
python train.py --setting=1 --batch-size=32 --patience=10 2>&1 | tee improve-set1.txt # hnd
python train.py --setting=combined 2>&1 | tee improve-setcom.txt
python train.py --setting=fedavg 2>&1 | tee improve-setfed.txt