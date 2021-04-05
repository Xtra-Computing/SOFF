#!/bin/bash

mkdir -p result_fed
python -u train_fed.py 2>&1 | tee -a result_fed/fed.log
