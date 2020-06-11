#!/bin/bash

# Epsilon-Accuracy
python train.py --dp -e=0.03125 
python train.py --dp -e=0.0625 
python train.py --dp -e=0.125 
python train.py --dp -e=0.25 
python train.py --dp -e=0.5 
python train.py --dp -e=1.0 
python train.py --dp -e=2.0 
python train.py --dp -e=4.0 
python train.py --dp -e=6.4 

# Lotsize-Accuracy: fedavg
python train.py --dp -e=2 --lotsize-scaler=0.1 
python train.py --dp -e=2 --lotsize-scaler=1 
python train.py --dp -e=2 --lotsize-scaler=3.1623 
python train.py --dp -e=2 --lotsize-scaler=10 
python train.py --dp -e=2 --lotsize-scaler=31.623 
python train.py --dp -e=2 --lotsize-scaler=100 
python train.py --dp -e=2 --lotsize-scaler=316.23 

# Lotsize-Accuracy: party 0 (CASIA-HWDB1.1)
python train.py --dp -e=2 --lotsize-scaler=0.1 --setting=0 
python train.py --dp -e=2 --lotsize-scaler=1 --setting=0 
python train.py --dp -e=2 --lotsize-scaler=3.1623 --setting=0 
python train.py --dp -e=2 --lotsize-scaler=10 --setting=0 
python train.py --dp -e=2 --lotsize-scaler=31.623 --setting=0 
python train.py --dp -e=2 --lotsize-scaler=100 --setting=0 
python train.py --dp -e=2 --lotsize-scaler=316.23 --setting=0 

# Lotsize-Accuracy: party 1 (HIT-OR3C)
python train.py --dp -e=2 --lotsize-scaler=0.1 --setting=1 
python train.py --dp -e=2 --lotsize-scaler=1 --setting=1 
python train.py --dp -e=2 --lotsize-scaler=3.1623 --setting=1 
python train.py --dp -e=2 --lotsize-scaler=10 --setting=1 
python train.py --dp -e=2 --lotsize-scaler=31.623 --setting=1 
python train.py --dp -e=2 --lotsize-scaler=100 --setting=1 
python train.py --dp -e=2 --lotsize-scaler=316.23 --setting=1 