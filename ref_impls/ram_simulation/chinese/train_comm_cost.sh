#!/bin/bash

mkdir -p result_comm_cost

for avgInterval in 1 2 3 4 5 6 7 8; do
    python -u ./train.py --local-epochs $avgInterval | tee -a "./result_comm_cost/result_fedavg_a_$avgInterval"
done