#!/bin/bash

python -u ./train.py --setting 0  # LA only
python -u ./train.py --setting 1  # BAY only
python -u ./train.py --setting combined
python -u ./train.py --setting fedavg

for epo in 2 3 4 5 6 7 8; do
    python -u ./train.py --local-epochs $epo
done

for epsilon in 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0 6.4; do
    python -u ./train.py -E 10 --dp -e $epsilon
done

for scaler in 0.1 1 3.1623 10 31.623 100 316.23; do
    python -u ./train.py -E 10 --lotsize-scaler $scaler --dp -e 2.0
done

for dataset in 0 1; do
    for scaler in 0.1 1 3.1623 10 31.623 100 316.23; do
        python -u ./train.py -E 10 --setting $dataset --lotsize-scaler $scaler --dp -e 2.0
    done
done