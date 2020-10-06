#!/bin/bash

# Communication cost
python train.py --local-epochs=2 2>&1 | tee comm-local2.txt
python train.py --local-epochs=3 2>&1 | tee comm-local3.txt
python train.py --local-epochs=4 2>&1 | tee comm-local4.txt
python train.py --local-epochs=5 2>&1 | tee comm-local5.txt
python train.py --local-epochs=6 2>&1 | tee comm-local6.txt
python train.py --local-epochs=7 2>&1 | tee comm-local7.txt
python train.py --local-epochs=8 2>&1 | tee comm-local8.txt