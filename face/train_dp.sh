#!/bin/bash

# Epsilon-Accuracy
# python train.py --epochs=20 2>&1 | tee epsacc-nondp.txt  # no need to run, see the result in improve-setfed.txt
python train.py --dp --epsilon=0.03125 --epochs=20 2>&1 | tee epsacc-eps0.03125.txt
python train.py --dp --epsilon=0.0625 --epochs=20 2>&1 | tee epsacc-eps0.0625.txt
python train.py --dp --epsilon=0.125 --epochs=20 2>&1 | tee epsacc-eps0.125.txt
python train.py --dp --epsilon=0.25 --epochs=20 2>&1 | tee epsacc-eps0.25.txt
python train.py --dp --epsilon=0.5 --epochs=20 2>&1 | tee epsacc-eps0.5.txt
python train.py --dp --epsilon=1.0 --epochs=20 2>&1 | tee epsacc-eps1.0.txt
python train.py --dp --epsilon=2.0 --epochs=20 2>&1 | tee epsacc-eps2.0.txt
python train.py --dp --epsilon=4.0 --epochs=20 2>&1 | tee epsacc-eps4.0.txt
python train.py --dp --epsilon=6.4 --epochs=20 2>&1 | tee epsacc-eps6.4.txt

# Lotsize-Accuracy: fedavg
python train.py --dp --epsilon=2.0 --epochs=20 --lotsize-scaler=0.1 2>&1 | tee lotacc-setfed-scale0.1.txt
# python train.py --dp --epsilon=2.0 --epochs=20 --lotsize-scaler=1.0 2>&1 | tee lotacc-setfed-scale1.0.txt  # no need to run, see the result in epsacc-eps2.0.txt
python train.py --dp --epsilon=2.0 --epochs=20 --lotsize-scaler=3.1623 2>&1 | tee lotacc-setfed-scale3.1623.txt
python train.py --dp --epsilon=2.0 --epochs=20 --lotsize-scaler=10.0 --patience=3 --patience-reducelr=1 2>&1 | tee lotacc-setfed-scale10.0.txt
python train.py --dp --epsilon=2.0 --epochs=20 --lotsize-scaler=31.623 2>&1 | tee lotacc-setfed-scale31.623.txt
python train.py --dp --epsilon=2.0 --epochs=20 --lotsize-scaler=100.0 2>&1 | tee lotacc-setfed-scale100.0.txt
python train.py --dp --epsilon=2.0 --epochs=20 --lotsize-scaler=316.23 2>&1 | tee lotacc-setfed-scale316.23.txt

# Lotsize-Accuracy: party 0
python train.py --dp --epsilon=2.0 --epochs=20 --setting=0 --lotsize-scaler=0.1 2>&1 | tee lotacc-set0-scale0.1.txt
python train.py --dp --epsilon=2.0 --epochs=20 --setting=0 --lotsize-scaler=1.0 2>&1 | tee lotacc-set0-scale1.0.txt
python train.py --dp --epsilon=2.0 --epochs=20 --setting=0 --lotsize-scaler=3.1623 2>&1 | tee lotacc-set0-scale3.1623.txt
python train.py --dp --epsilon=2.0 --epochs=20 --setting=0 --lotsize-scaler=10.0 2>&1 | tee lotacc-set0-scale10.0.txt
python train.py --dp --epsilon=2.0 --epochs=20 --setting=0 --lotsize-scaler=31.623 2>&1 | tee lotacc-set0-scale31.623.txt
python train.py --dp --epsilon=2.0 --epochs=20 --setting=0 --lotsize-scaler=100.0 2>&1 | tee lotacc-set0-scale100.0.txt
python train.py --dp --epsilon=2.0 --epochs=20 --setting=0 --lotsize-scaler=316.23 2>&1 | tee lotacc-set0-scale316.23.txt

# Lotsize-Accuracy: party 1
python train.py --dp --epsilon=2.0 --epochs=20 --setting=1 --lotsize-scaler=0.1 2>&1 | tee lotacc-set1-scale0.1.txt
python train.py --dp --epsilon=2.0 --epochs=20 --setting=1 --lotsize-scaler=1.0 2>&1 | tee lotacc-set1-scale1.0.txt
python train.py --dp --epsilon=2.0 --epochs=20 --setting=1 --lotsize-scaler=3.1623 2>&1 | tee lotacc-set1-scale3.1623.txt
python train.py --dp --epsilon=2.0 --epochs=20 --setting=1 --lotsize-scaler=10.0 2>&1 | tee lotacc-set1-scale10.0.txt
python train.py --dp --epsilon=2.0 --epochs=20 --setting=1 --lotsize-scaler=31.623 2>&1 | tee lotacc-set1-scale31.623.txt
python train.py --dp --epsilon=2.0 --epochs=20 --setting=1 --lotsize-scaler=100.0 2>&1 | tee lotacc-set1-scale100.0.txt
python train.py --dp --epsilon=2.0 --epochs=20 --setting=1 --lotsize-scaler=316.23 2>&1 | tee lotacc-set1-scale316.23.txt