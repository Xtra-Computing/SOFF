#!/bin/bash

python -um soff.launcher.client_server +C "MLP - SmokingDrinking: SplitNN" \
    +a splitnn +n 2 \
    +sa -tr.e 200 -tr.bs 1024 -tr.s.n multistep -tr.s.mltstp.mlstns 50 100 150 \
        -tr.lr 0.2 -tr.o.n sgd \
        -tr.m.ns baccuracy bauroc \
        -dt.fs.m vertical-subsample -dt.fs.v.sr 0.1 -dt.fs.n 2 -dt.r.d splitnn-smoking-drinking-labels \
        -md.n splitnn-binary-mlp-server -md.spnn.i 40 -md.spnn.sh 20 -md.l.n bcewl \
    +ca -md.spnn.o 20  -md.spnn.ch 20  \
        -dt.fs.m vertical-subsample -dt.fs.v.sr 0.1 -dt.fs.n 2 \
    +pca -0 -md.n splitnn-binary-mlp-client -dt.r.d splitnn-smoking-drinking-features1 \
        -1 -md.n splitnn-binary-mlp-client -dt.r.d splitnn-smoking-drinking-features2


