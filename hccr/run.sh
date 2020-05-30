#!/bin/bash

# put data/HWDB1.1fullset.hdf5, data/HIT_OR3Cfullset.hdf5, data/HIT_HWDB1.1_fullset.hdf5

# experiment 1
python hccr.py --setting=casia
python hccr.py --setting=hit
python hccr.py --setting=combined
python hccr.py --setting=fedavg

# experiment 2
python hccr.py --freq=1
python hccr.py --freq=2
python hccr.py --freq=3
python hccr.py --freq=4
python hccr.py --freq=5
python hccr.py --freq=6
python hccr.py --freq=7
python hccr.py --freq=8

# experiment 3
python hccr.py --dp -e=0.03125
python hccr.py --dp -e=0.0625
python hccr.py --dp -e=0.125
python hccr.py --dp -e=0.25
python hccr.py --dp -e=0.5
python hccr.py --dp -e=1.0
python hccr.py --dp -e=2.0
python hccr.py --dp -e=4.0

# experiment 3.1
python fl.py --dp -e=2 --lotsize-scaler=0.1
python fl.py --dp -e=2 --lotsize-scaler=1
python fl.py --dp -e=2 --lotsize-scaler=3.1623
python fl.py --dp -e=2 --lotsize-scaler=10
python fl.py --dp -e=2 --lotsize-scaler=31.623
python fl.py --dp -e=2 --lotsize-scaler=100
python fl.py --dp -e=2 --lotsize-scaler=316.23

python fl.py --dp -e=2 --lotsize-scaler=0.1 --setting=casia
python fl.py --dp -e=2 --lotsize-scaler=1 --setting=casia
python fl.py --dp -e=2 --lotsize-scaler=3.1623 --setting=casia
python fl.py --dp -e=2 --lotsize-scaler=10 --setting=casia
python fl.py --dp -e=2 --lotsize-scaler=31.623 --setting=casia
python fl.py --dp -e=2 --lotsize-scaler=100 --setting=casia
python fl.py --dp -e=2 --lotsize-scaler=316.23 --setting=casia

python fl.py --dp -e=2 --lotsize-scaler=0.1 --setting=hit
python fl.py --dp -e=2 --lotsize-scaler=1 --setting=hit
python fl.py --dp -e=2 --lotsize-scaler=3.1623 --setting=hit
python fl.py --dp -e=2 --lotsize-scaler=10 --setting=hit
python fl.py --dp -e=2 --lotsize-scaler=31.623 --setting=hit
python fl.py --dp -e=2 --lotsize-scaler=100 --setting=hit
python fl.py --dp -e=2 --lotsize-scaler=316.23 --setting=hit
