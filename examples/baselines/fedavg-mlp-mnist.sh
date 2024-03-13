#!/bin/bash

python -um soff.launcher.client_server +C "Alexnet - MNIST: Fedavg " \
    +a fedavg +n 4 +sa -md.n mnist-mlp -dt.r.d mnist \
    -tr.mt.n per-iter -fedavg.a 1 -fl.te 50 \
    -tr.bs 32 -tr.s.n constant -tr.s.mltstp.mlstns 200 300 \
    -tr.o.sgd.wghtdcy 1e-3 \
    -tr.e 400 -tr.lr 0.1
