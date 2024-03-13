#!/bin/bash

python -um soff.launcher.client_server +C "HybridLSTM - Abchina: Fedavg " \
    +envs CUDA_LAUNCH_BLOCKING=1 +a ff_fedavg +n 4 +sa \
    -md.n hybrid-cnn-lstm -dt.r.d abchina -dt.r.abc.ysl 300 -dt.r.abc.zsl 300 \
    -md.l.n bce -md.hlstm.dp 0.0 -dc/s.rl 12_500_000 \
    -tr.mt.n hybrid-lstm-per-epoch -fedavg.a 1 -fl.te 1 \
    -tr.o.n sgd -tr.m.ns baccuracy bauroc \
    -tr.bs 128 -tr.s.n multistep -tr.s.mltstp.mlstns 70 140  \
    -tr.o.sgd.wghtdcy 1e-5 \
    -tr.e 200 -tr.lr 1.0

# rate limit
# md.l.n bce -md.hlstm.dp 0.0 -dc.rl 12_500_000 \
