#!/bin/bash

trainSets=("imdb" "amazon" "both")

mkdir -p result

i=0
#for i in $(seq 1 5); do
for trainSet in "${trainSets[@]}"; do
    echo "== ${trainSet} $i ==========================================================================="
    python -u ./sentiment.py "${trainSet}" | tee "./result/result_${trainSet}_$i"
done

echo "== Federated $i ============================================================================="
python -u ./sentiment-federated-simulated.py | tee -a "./result/result_fedavg_$i"
#done
