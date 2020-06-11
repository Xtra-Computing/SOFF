#!/bin/bash

mkdir -p result_dp

for epsilon in 0.03125 0.05 0.1 0.2 0.4 0.8 1.6 3.2 6.4; do
    python -u ./sentiment-federated-dp.py -E 20 -l 5 -e $epsilon | tee -a "./result_dp/epochs_20_batchsize_40_bpl_5_e_""$epsilon"
done

for bpl in 1 2 3 5 10 20 40 80 160 320 640; do
    python -u ./sentiment-federated-dp.py -E 20 -l $bpl -e 2.0 | tee -a "./result_dp/epochs_20_batchsize_40_bpl_""$bpl""_e_2.0"
done

mkdir -p result_dp_single

for dataset in imdb amazon; do
    for bpl in 1 2 3 5 10 20 40 80 160 320 640; do
        python -u ./sentiment-dp.py -E 20 -D $dataset -l $bpl -e 2.0 | tee -a "./result_dp_single/""$dataset""_batchsize_40_bpl_""$bpl""_e_2.0"
    done
done
