#!/bin/bash

mkdir -p result/dp-dual
for lotsize_scaler in 0.05 1.0 2.0 4.0 8.0 16 32; do
    python -u age-federated-dp.py -E 30 -sc -ep 2.0 -ls $lotsize_scaler | tee result/dp-dual/result_fedavg_allagefaces_appa_epsilon_2.0_lotsize_$lotsize_scaler
done
