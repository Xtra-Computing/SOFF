#!/bin/bash

for setting in A B; do
python -um soff.launcher.client_server +C "MLP - ECommerce: SplitNN" \
    +a splitnn +n 2 \
    +sa -fl.sd 1 -tr.e 20 -tr.bs 1024 -tr.s.n multistep -tr.s.mltstp.mlstns 10 15 \
        -tr.lr 0.1 -tr.o.n sgd -tr.m.ns baccuracy bauroc \
        -dt.fs.m vertical -dt.fs.n 2 \
        -dt.r.d splitnn-ecommerce-labels -dt.r.ecmm.os -dt.r.ecmm.st $setting \
        -md.n splitnn-binary-mlp-server -md.spnn.i 4000 -md.spnn.sh 2000 \
        -md.l.n bcewl \
    +ca -fl.sd 1 -md.spnn.o 2000 -md.spnn.ch 2000 \
        -dt.fs.m vertical -dt.fs.n 2 -dt.r.ecmm.os -dt.r.ecmm.st $setting \
    +pca -0 -md.n splitnn-binary-mlp-client -dt.r.d splitnn-ecommerce-features1 \
        -1 -md.n splitnn-binary-mlp-client -dt.r.d splitnn-ecommerce-features2
    break
done
