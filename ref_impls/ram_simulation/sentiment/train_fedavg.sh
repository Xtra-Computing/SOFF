#!/bin/bash

trainSets=("imdb" "amazon" "both")

mkdir -p result_fedavg

#for i in $(seq 1 5); do
for trainSet in "${trainSets[@]}"; do
    python -u ./sentiment.py "${trainSet}" | tee "./result_fedavg/result_${trainSet}"
done

python -u ./sentiment-federated.py | tee -a "./result_fedavg/result_fedavg"
