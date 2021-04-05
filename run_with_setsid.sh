#!/bin/bash

notify() {
    if command -v gotify-cli 2>&1; then
        gotify-cli push -t "OARF: $(hostname)" "run $* finished"
    fi
}

_get_session_name() {
    while [[ $# -gt 0 ]]; do
        arg="$1"
        shift
        case $arg in
        '+sn' | '++session-name')
            session_name=$1
            shift
            ;;
        esac
    done
    echo "${session_name:-comm}"
}

ADDITIONAL_ARGS=()

run_one_task() {
    session_name="$(_get_session_name "$@")"
    sync_file="${session_name}.lock"

    ./run.sh "${ADDITIONAL_ARGS[@]}" "$@" >>_error.log 2>&1

    # wait for this session to stop
    while tmux has-session -t "$session_name" >/dev/null 2>&1; do
        sleep 1
    done

    # wait for file
    while ! [[ -f "$sync_file" ]]; do
        sleep 1
    done

    notify "$@"

    rm "$sync_file"
}

SEED=${SEED:-0}

# 0. Utilities ================================================================
run_3_rounds() {
    for i in $(seq 0 3); do
        SEED="$i"
        for cmd in "$@"; do
            "$cmd"
        done
    done
}

run_with_seed() {
    SEED="$1"
    shift
    for cmd in "$@"; do
        "$cmd"
    done
}

run_with_seeds() {
    seeds=()
    while [[ $# -gt 0 && $1 != '--' ]]; do
        seeds+=("$1")
        shift
    done

    if [[ $1 == '--' ]]; then
        shift
    elif [[ $# -eq 0 ]]; then
        echo "No command to run :("
        return 1
    fi

    for seed in "${seeds[@]}"; do
        SEED="$seed"
        for cmd in "$@"; do
            "$cmd"
        done
    done
}

run_with_sdees() {
    seeds=()
    while [[ $# -gt 0 && $1 != '--' ]]; do
        seeds+=("$1")
        shift
    done

    if [[ $1 == '--' ]]; then
        shift
    elif [[ $# -eq 0 ]]; then
        echo "No command to run :("
        return 1
    fi

    for cmd in "$@"; do
        for seed in "${seeds[@]}"; do
            SEED="$seed"
            "$cmd"
        done
    done
}

# 1. Baselines ================================================================
baseline() {
    baseline_single
    baseline_fedsgd
}

# Single-dataset baseline -----------------------------------------------------
baseline_single() {
    baseline_single_sentiment
    baseline_single_face_age
}

baseline_single_cifar10() {
    run_one_task +sn cifar10-seed-"$SEED" +a fedsgd +n 1 +sa -sd "$SEED" -bt model -d cifar10 -dsr 0.2
}

baseline_single_sentiment() {
    run_one_task +sn imdb-seed-"$SEED" +a fedsgd +n 1 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn amaz-seed-"$SEED" +a fedsgd +n 1 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:amazon -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
}

baseline_single_face_age() {
    run_one_task +sn allage-seed-"$SEED" +a fedsgd +n 1 +sa -sd "$SEED" -bs 32 -bt model -d face:allagefaces -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn appa-seed-"$SEED" +a fedsgd +n 1 +sa -sd "$SEED" -bs 32 -bt model -d face:appa -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn wiki-seed-"$SEED" +a fedsgd +n 1 +sa -sd "$SEED" -bs 32 -bt model -d face:wiki -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn utk-seed-"$SEED" +a fedsgd +n 1 +sa -sd "$SEED" -bs 32 -bt model -d face:utk -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
}

# FedSGD baseline -------------------------------------------------------------
baseline_fedsgd() {
    baseline_fedsgd_cifar10
    baseline_fedsgd_sentiment
    baseline_fedsgd_face_age
}

baseline_fedsgd_cifar10() {
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10
}

baseline_fedsgd_sentiment() {
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
}

baseline_fedsgd_face_age() {
    run_one_task +sn face-fedsgd-seed-"$SEED" +a fedsgd +n 4 +sa -sd "$SEED" -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
}

# Combined baseline -----------------------------------------------------------
baesline_combined() {
    baseline_combined_cifar10
    baseline_combined_sentiment
    baseline_combined_face_age
}

baseline_combined_cifar10() {
    run_one_task +sn cifar10-comb-seed-"$SEED" +a fedsgd +n 1 +sa -sd "$SEED" -bt model -d cifar10
}

baseline_combined_sentiment() {
    run_one_task +sn sent-comb-seed-"$SEED" +a fedsgd +n 1 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
}

baseline_combined_face_age() {
    run_one_task +sn face-comb-seed-"$SEED" +a fedsgd +n 1 +sa -sd "$SEED" -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
}

# 2. Non-i.i.d-ness ===========================================================
noniid() {
    noniid_dirichlet
    noniid_powerlaw
}

noniid_dirichlet() {
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -ds quantity-skew-dirichlet -al 0.2
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -ds quantity-skew-dirichlet -al 0.4
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -ds quantity-skew-dirichlet -al 0.6
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -ds quantity-skew-dirichlet -al 0.8
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -ds quantity-skew-dirichlet -al 1.0

    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -ds label-skew-dirichlet -al 0.2
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -ds label-skew-dirichlet -al 0.4
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -ds label-skew-dirichlet -al 0.6
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -ds label-skew-dirichlet -al 0.8
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -ds label-skew-dirichlet -al 1.0
}

# noniid_powerlaw() {
#     run_one_task +a fedsgd +n 10 +sa -bt model -d cifar10 -ds powerlaw -al 0.2
#     run_one_task +a fedsgd +n 10 +sa -bt model -d cifar10 -ds powerlaw -al 0.4
#     run_one_task +a fedsgd +n 10 +sa -bt model -d cifar10 -ds powerlaw -al 0.6
#     run_one_task +a fedsgd +n 10 +sa -bt model -d cifar10 -ds powerlaw -al 0.8
#     run_one_task +a fedsgd +n 10 +sa -bt model -d cifar10 -ds powerlaw -al 1.0
# }

noniid_sentiment() {
    :
}

# 3. Algorithm ================================================================
algorithm_cifar10() {
    run_one_task +sn cifar10-fedavg-seed-"$SEED" +a fedavg +n 5 +sa -sd "$SEED" -e 2000 -cf 0.5 -bt model -d cifar10
    run_one_task +sn cifar10-fednova-seed-"$SEED" +a fednova +n 5 +sa -sd "$SEED" -e 2000 -cf 0.5 -bt model -d cifar10
    run_one_task +sn cifar10-fedprox-seed-"$SEED" +a fedprox +n 5 +sa -sd "$SEED" -e 2000 -cf 0.5 -mu 0.01 -bt model -d cifar10
    run_one_task +sn cifar10-fedprox-seed-"$SEED" +a fedprox +n 5 +sa -sd "$SEED" -e 2000 -cf 0.5 -mu 0.001 -bt model -d cifar10
}

algorithm_sentiment() {
    run_one_task +sn sent-fedavg-seed-"$SEED" +a fedavg +n 2 +sa -sd "$SEED" -e 2000 -cf 0.5 -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedprox-seed-"$SEED" +a fedprox +n 2 +sa -sd "$SEED" -e 2000 -cf 0.5 -mu 0.01 -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedprox-seed-"$SEED" +a fedprox +n 2 +sa -sd "$SEED" -e 2000 -cf 0.5 -mu 0.001 -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fednova-seed-"$SEED" +a fednova +n 2 +sa -sd "$SEED" -e 2000 -cf 0.5 -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
}

algorithm_face_age() {
    run_one_task +sn face-fedavg-seed-"$SEED" +a fedavg +n 4 +sa -sd "$SEED" -cf 0.5 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn face-fedprox-seed-"$SEED" +a fedprox +n 4 +sa -sd "$SEED" -cf 0.5 -mu 0.01 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn face-fedprox-seed-"$SEED" +a fedprox +n 4 +sa -sd "$SEED" -cf 0.5 -mu 0.001 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn face-fednova-seed-"$SEED" +a fednova +n 4 +sa -sd "$SEED" -cf 0.5 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn face-fedprox-seed-"$SEED" +a fedprox +n 4 +sa -sd "$SEED" -cf 0.5 -mu 0.01 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
}

# 4. Privacy ==================================================================
privacy_cifar10() {
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -dt rdp -cl 0.1 -ep 16.0
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -dt rdp -cl 0.1 -ep 8.0
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -dt rdp -cl 0.1 -ep 4.0
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -dt rdp -cl 0.1 -ep 2.0
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -dt rdp -cl 0.1 -ep 1.0
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -dt rdp -cl 0.1 -ep 0.5
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -dt rdp -cl 0.1 -ep 0.25
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -dt rdp -cl 0.1 -ep 0.125
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -dt rdp -cl 0.1 -ep 0.0625
}

privacy_sentiment() {
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 16.0 -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 8.0 -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 4.0 -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 2.0 -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 1.0 -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 0.5 -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 0.25 -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 0.125 -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 0.0625 -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
}

privacy_face_age() {
    run_one_task +sn face-fedsgd-seed-"$SEED" +a fedsgd +n 4 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 16.0 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn face-fedsgd-seed-"$SEED" +a fedsgd +n 4 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 8.0 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn face-fedsgd-seed-"$SEED" +a fedsgd +n 4 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 4.0 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn face-fedsgd-seed-"$SEED" +a fedsgd +n 4 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 2.0 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn face-fedsgd-seed-"$SEED" +a fedsgd +n 4 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 1.0 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn face-fedsgd-seed-"$SEED" +a fedsgd +n 4 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 0.5 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn face-fedsgd-seed-"$SEED" +a fedsgd +n 4 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 0.25 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn face-fedsgd-seed-"$SEED" +a fedsgd +n 4 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 0.125 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
    run_one_task +sn face-fedsgd-seed-"$SEED" +a fedsgd +n 4 +sa -sd "$SEED" -dt rdp -cl 0.1 -ep 0.0625 -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
}

# 5. Encryption ===============================================================
encryption_cifar10() {
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -sam SS -ssn 2
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -sam SS -ssn 4
}

encryption_sentiment() {
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 -sam SS -ssn 2 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
}

encryption_face_age() {
    run_one_task +sn face-fedsgd-seed-"$SEED" +a fedsgd +n 4 +sa -sd "$SEED" -bs 32 -bt model -d face:allagefaces face:appa face:wiki face:utk -ds realistic -M VGG16 -lr 0.01 -sam SS -ssn 2 \
        -td face:allagefaces face:appa face:wiki face:utk face:allagefaces,face:appa,face:wiki,face:utk
}

# 6. Compression ==============================================================
compression_cifar10() {
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -a 2
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -a 4
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -a 6
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -a 8
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -a 10
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -cc topk_permodel -cr 0.003
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -cc randk_permodel
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 -cc rankk_perlayer -ck 3 -ef -ed 0.5
}

compression_sentiment() {
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 -a 2 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 -a 4 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 -a 6 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 -a 8 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 -a 10 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 -cc topk_permodel -cr 0.003 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 -cc randk_permodel \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 -cc rankk_perlayer -ck 3 -ef -ed 0.5 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
}

# 7. Hybrid ===================================================================

hybrid_baseline_cifar10() {
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 \
        -rl 12_500_000
}

hybrid_cifar10() {
    run_one_task +sn cifar10-fedsgd-seed-"$SEED" +a fedsgd +n 5 +sa -sd "$SEED" -bt model -d cifar10 \
        -sam SS -ssn 2 -dt rdp -cl 0.1 -ep 2.0 -rl 12_500_000 -a 4 -cc topk_permodel -cr 0.01
}

hybrid_baseline_sentiment() {
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -rl 12_500_000 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
}

hybrid_sentiment() {
    run_one_task +sn sent-fedsgd-seed-"$SEED" +a fedsgd +n 2 +sa -sd "$SEED" -we 0 -p 20 -bt model -d sentiment:imdb sentiment:amazon -ds realistic -M LSTM -o Adam -lr 0.001 \
        -sam SS -ssn 2 -dt rdp -cl 0.1 -ep 2.0 -rl 12_500_000 -a 4 -cc topk_permodel -cr 0.01 \
        -td sentiment:imdb sentiment:amazon sentiment:imdb,sentiment:amazon
}

hybrid_face_age() {
    :
}

"$@"

notify "ALL: $*"
