#!/bin/bash

pushd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../.. || exit 1

LOG_DIR=log
log_dir="interop_test"
mkdir -p "$LOG_DIR/$log_dir"

NUM_CLIENTS=4
GPUID=0

env PYTHONPATH=. python -um soff.launcher.node.node \
    -l.a fedavg.server \
    -cs.n "$NUM_CLIENTS" -ss.n "$NUM_CLIENTS" -dt.fs.n "$NUM_CLIENTS" \
    -hw.gs "$GPUID" \
    -dc.s tcp -dc.a 0.0.0.0:18030 \
    -md.n resnet18 -dt.r.d cifar10 \
    -tr.mt.n per-epoch -fedavg.a 1 -fl.te 1 \
    -tr.bs 32 -tr.s.n constant \
    -tr.o.sgd.wghtdcy 1e-3 \
    -tr.e 400 -tr.lr 0.1 \
    -lg.f "$LOG_DIR/$log_dir/server.log" \
    -lg.df "$LOG_DIR/$log_dir/server.csv" \
    -lg.tld "$LOG_DIR/$log_dir/tfboard" \
    2>"$LOG_DIR/$log_dir/server.err"

popd  || exit 1
