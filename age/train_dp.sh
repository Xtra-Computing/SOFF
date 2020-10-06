#!/bin/bash

mkdir -p result/dp
#for eps in 0.05 0.1 0.2 0.4 0.8 1.6 3.2 6.4; do
for eps in                              6.4; do
    python -u age-federated-dp.py -E 30 -sc -ep $eps | tee result/dp/result_fedavg_allagefaces_appa_epsilon_$eps
done
