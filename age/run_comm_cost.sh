#!/bin/bash

mkdir -p result/comm_cost

for i in 1 2 3 4 5 6 7 8 9 10; do
    python -u age-federated.py -sc -a $i | tee result/comm_cost/result_fedavg_allagefaces_appa_every_$i
done
