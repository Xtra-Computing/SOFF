#!/bin/bash

python -um soff.launcher.client_server +C "Resnet18 - CIFAR10: Fedavg Baseline" \
    +a fedavg +n 4 +sa -md.n resnet18 -dt.r.d cifar10 \
    -tr.mt.n per-iter -fedavg.a 1 -fl.te 50 \
    -tr.bs 128 -tr.s.n multistep -tr.s.mltstp.mlstns 200 300 \
    -tr.o.sgd.mmntm 0.9 -tr.o.sgd.wghtdcy 1e-3 \
    -tr.e 400 -tr.lr 0.1
