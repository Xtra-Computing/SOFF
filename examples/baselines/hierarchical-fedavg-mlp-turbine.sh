#!/bin/bash

python -um soff.launcher.hierarchical +C "Alexnet - Turbine: Hierarchical Fedavg " \
    +a h_fedavg +hi "(7, 6, 7, 6, 7)" \
    +sa -dt.r.d turbine -dt.fs.m delegated  \
    -md.n turbine-mlp -md.l.n bce \
    -tr.m.ns bauroc baccuracy \
    -tr.mt.n per-epoch -fedavg.a 1 -fl.te 1 \
    -tr.bs 128 -tr.s.n multistep -tr.s.mltstp.mlstns 100 200 \
    -tr.o.n sgd -tr.o.sgd.wghtdcy 1e-5 \
    -tr.e 300 -tr.lr 1.0 \
    -dc.rl 12500000

# python -um soff.launcher.client_server +C "Alexnet - Turbine: Centralized Fedavg" \
#     +a fedavg +n 33 \
#     +sa -dt.r.d turbine -dt.fs.m delegated  \
#     -md.n turbine-mlp -md.l.n bce \
#     -tr.m.ns bauroc baccuracy \
#     -tr.mt.n per-epoch -fedavg.a 1 -fl.te 1 \
#     -tr.bs 128 -tr.s.n multistep -tr.s.mltstp.mlstns 100 200 \
#     -tr.o.n sgd -tr.o.sgd.wghtdcy 1e-5 \
#     -tr.e 300 -tr.lr 1.0 \
#     -dc.rl 12500000
