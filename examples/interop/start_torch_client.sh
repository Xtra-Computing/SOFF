#!/bin/bash

pushd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../.. || exit 1

LOG_DIR=log
log_dir="interop_test"
mkdir -p "$LOG_DIR/$log_dir"

GPUID=0

client_id=0
while [[ -f "$LOG_DIR/$log_dir/client$client_id.err" ]]; do ((++client_id)); done

env PYTHONPATH=. python -um soff.launcher.node.node \
    -l.a fedavg.client \
    -hw.gs "$GPUID" \
    -dc.s tcp -dc.a 127.0.0.1:18030 \
    -lg.f "$LOG_DIR/$log_dir/client$client_id.log" \
    -lg.df "$LOG_DIR/$log_dir/client$client_id.csv" \
    -lg.tld "$LOG_DIR/$log_dir/tfboard" \
    2>"$LOG_DIR/$log_dir/client$client_id.err"

popd  || exit 1
