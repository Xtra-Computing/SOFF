#!/bin/bash

mkdir -p results
python -u songs.py msd -o adam -lr 0.01 | tee results_msd
python -u songs.py fma -o adam -lr 0.01 | tee results_fma
python -u songs.py union -o adam -lr 0.01 | tee results_union
python -u songs.py union -o adam -lr 0.01 -sn | tee results_union_federated

# run multiple times as sometime it does not converge to optimal
for i in {1..5}; do
    python -u songs.py msd_aligned -o adam -lr 0.1 | tee results_msd_aligned_"$i"
    python -u songs.py fma_aligned -o adam -lr 0.1 | tee results_fma_aligned_"$i"
    python -u songs.py aligned -o adam -lr 0.1 | tee results_aligned_"$i"
    python -u songs.py aligned -o adam -lr 0.1 -sn | tee results_aligned_federated_"$i"
done
