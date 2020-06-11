#!/bin/bash

mkdir -p result_dp result_dp_single

# Epsilon-Accuracy
for epsilon in 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0 6.4; do
    python -u ./train.py -E 10 --dp -e $epsilon | tee -a "./result_dp/epochs_10_batchsize_128_scaler_1_e_""$epsilon"
done

# Lotsize-Accuracy: fedavg
for scaler in 0.1 1 3.1623 10 31.623 100 316.23; do
    python -u ./train.py -E 10 --lotsize-scaler $scaler --dp -e 2.0 | tee -a "./result_dp/epochs_10_batchsize_128_scaler_""$scaler""_e_2.0"
done

# Lotsize-Accuracy: single party
# party 0 (CASIA-HWDB1.1)
# party 1 (HIT-OR3C)
for dataset in 0 1; do
    for scaler in 0.1 1 3.1623 10 31.623 100 316.23; do
        python -u ./train.py -E 10 --setting $dataset --lotsize-scaler $scaler --dp -e 2.0 | tee -a "./result_dp_single/""$dataset""_batchsize_128_scaler_""$scaler""_e_2.0"
    done
done