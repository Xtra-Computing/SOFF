#!/bin/bash
# Copyright (c) Hu Sixu <husixu1@hotmail.com>. All rights reserved.

readonly invoke_string="$(printf %q "${BASH_SOURCE[0]}")$([[ -n $# ]] && printf ' %q' "$@")"

# Parse parameters ############################################################
no_tfboard=false
no_log=false
server_args=()
client_args=()
while [[ $# -gt 0 ]]; do
    arg="$1"
    shift
    case $arg in
    '+a' | '++algorithm')
        algorithm=$1
        shift
        ;;
        # use + instead of - to avoid conflict with python arguments
    '+sn' | '++session-name')
        session_name=$1
        shift
        ;;
    '+s' | '++socket-file')
        socket_type='unix'
        socket_file=$1
        shift
        ;;
    '+t' | '++tcp-addr')
        socket_type='tcp'
        tcp_addr=$1
        shift
        ;;
    '+n' | '++num-clients')
        readonly num_clients=$1
        shift
        if ! [[ $num_clients =~ [[:digit:]]+ ]]; then
            echo "num clients must be an integer" >&2
            exit 1
        fi
        ;;
    '+c' | '++run-config')
        runconfig=$1
        shift
        if ! [[ -f $runconfig ]]; then
            echo "runconfig must be a file" >&2
            exit 1
        fi
        ;;
    '+l' | '++log-dir')
        log_dir=$1
        shift
        ;;
    '+ntfb' | '++no-tfboard') # disable tensor-board, for debug
        no_tfboard=true
        ;;
    '+nl' | '++no-log') # disable log, for debug
        no_log=true
        ;;
    '+sa' | '++server-arguments')
        while [[ $# -gt 0 ]] && ! [[ $1 == +* ]]; do
            server_args+=("$1")
            shift
        done
        ;;
    '+ca' | '++client-arguments')
        while [[ $# -gt 0 ]] && ! [[ $1 == +* ]]; do
            client_args+=("$1")
            shift
        done
        ;;
    esac
done

algorithm="${algorithm:-fedsgd}"
case "$algorithm" in
'fedsgd')
    py_module="oarf.algorithms.fedsgd"
    py_server_class="FedSGDServer"
    py_client_class="FedSGDClient"
    ;;
'fedavg')
    py_module="oarf.algorithms.fedavg"
    py_server_class="FedAvgServer"
    py_client_class="FedAvgClient"
    ;;
'fedprox')
    py_module="oarf.algorithms.fedprox"
    py_server_class="FedProxServer"
    py_client_class="FedProxClient"
    ;;
'fednova')
    py_module="oarf.algorithms.fednova"
    py_server_class="FedNovaServer"
    py_client_class="FedNovaClient"
    ;;
esac

# Config #######################################################################
set -eo pipefail

session_name=${session_name:-comm}
sync_file="${session_name}.lock"
socket_type="${socket_type:-unix}"
socket_file=${socket_file:-/tmp/fed-${session_name}.sock}
# TODO: make tcp address for different sessions unique
runconfig=${runconfig:-runconfig.txt}

if [[ $socket_type == unix ]]; then
    server_addr_options="-s unix -ad $socket_file"
    client_addr_options="-s unix -ad $socket_file"
elif [[ $socket_type == tcp ]]; then
    server_addr_options="-s tcp -ad 0.0.0.0:${tcp_addr#*:}"
    client_addr_options="-s tcp -ad $tcp_addr"
fi

server_host=
server_gpu=
hosts=()
gpu=()
i=0
while read -r line; do
    read -ra linearray <<<"$line"
    # ignore comment lines
    [[ $line =~ ^[[:space:]]*# ]] && continue

    [[ "${#linearray[@]}" -ne 3 ]] && {
        echo "Unidentified line: $line"
        echo "Each line in $runconfig should be one of the following: "
        echo "  # <comments>"
        echo "  server <server-addr> <gpu-index>"
        echo "  client <server-addr> <gpu-index>"
        exit 1
    }

    # client config
    [[ ${linearray[0]} =~ ^c.* ]] && {
        hosts+=("${linearray[1]}")
        gpu+=("${linearray[2]}")
    }

    # server config
    [[ ${linearray[0]} =~ ^s.* ]] && {
        server_host="${linearray[1]}"
        server_gpu="${linearray[2]}"
    }
    ((++i))
done <"$runconfig"

if [[ ${#hosts[@]} -lt $num_clients ]]; then
    echo "config file does not contain sufficient lines" >&2
    echo "no_sufficnet_line" >"$sync_file"
    exit 1
fi

# first create a new session
date=$(date +%F_%H-%M-%S)
log_dir=${log_dir:-$date}

if ! $no_log; then
    mkdir -p "log/$log_dir"
    for host in "${hosts[@]}"; do
        ssh "$host" "cd $(pwd); mkdir -p \"log/$log_dir\""
    done
fi
if ! $no_tfboard; then
    mkdir -p "log/__tfboard/$log_dir";
    for host in "${hosts[@]}"; do
        ssh "$host" "cd $(pwd); mkdir -p \"log/__tfboard/$log_dir\""
    done
fi

# store the calling string for reproducability proposes
if ! $no_log; then echo "$invoke_string" >"log/$log_dir/invoke_command.txt"; fi

# Server #######################################################################
gen_server_command() {
    cmd=""
    cmd+=" cd $(pwd); "
    cmd+=" env CUDA_VISIBLE_DEVICES=${server_gpu} "
    cmd+=" python3 -u -c 'from $py_module import ${py_server_class}; ${py_server_class}.start();' "
    cmd+=" --num-clients $num_clients "
    cmd+=" ${server_addr_options} "
    cmd+=" ${server_args[*]} "
    if ! $no_log; then cmd+=" --log-file=log/$log_dir/server.log "; fi
    if ! $no_tfboard; then cmd+=" --tensorboard-log-dir log/__tfboard/$log_dir "; fi
    if ! $no_log; then cmd+=" 2>\"log/$log_dir/server.err\" "; fi
    echo -n "$cmd"
}

# shellcheck disable=SC2016
deffunc='settitle() {
      export PS1="\[\e[32m\]\u@\h \[\e[33m\]\w\[\e[0m\]\n$ "
      echo -ne "\e]0; $1 \a"
}'

# Start server
echo "Starting server"
export SHELL=/bin/bash
tmux new-session -d -s "$session_name" -n group-0 \
    "$deffunc; \
    settitle server; \
    ssh -t ${server_host} \"$(gen_server_command)\"; \
    echo server_done >$sync_file; "

tmux set-option -g -t "$session_name" pane-border-status top
tmux set-option -g -t "$session_name" pane-border-format "#T"

# Clients ######################################################################
# wait for server to start

sleepcount=0
wait_or_kill() {
    sleep 1
    ((++sleepcount))
    if [[ $sleepcount -gt 300 ]]; then
        echo "wait_timout" >"${sync_file}"
        exit 1
    fi
}

if [[ $socket_type == unix ]]; then
    while ! [[ -S ${socket_file} ]]; do
        wait_or_kill
    done
elif [[ $socket_type == tcp ]]; then
    while ! nc -zv "${tcp_addr%:*}" "${tcp_addr#*:}" >/dev/null 2>&1; do
        wait_or_kill
    done
fi
sleep 5

window=0

client_id=0

# $1: client_id
gen_client_command() {
    cmd=""
    cmd+=" cd $(pwd); "
    cmd+=" env CUDA_VISIBLE_DEVICES=${gpu[$1]}"
    cmd+=" python3 -u -c 'from ${py_module} import ${py_client_class}; ${py_client_class}.start();' "
    cmd+=" ${client_addr_options} "
    cmd+=" ${client_args[*]} "
    if ! $no_log; then cmd+=" --log-file=log/$log_dir/client$1.log "; fi
    if ! $no_tfboard; then cmd+=" --tensorboard-log-dir log/__tfboard/$log_dir "; fi
    if ! $no_log; then cmd+=" 2> \"log/$log_dir/client$1.err\" "; fi
    echo -n "$cmd"
}

num_clients_remain="$num_clients"
while ((num_clients_remain > 0)); do
    if [[ $window == 0 ]]; then num_pane=5; else num_pane=6; fi

    # decide total number of panes in this window
    if ((num_clients_remain < num_pane)); then
        num_pane=$num_clients_remain
        if [[ $window == 0 ]]; then ((++num_pane)); fi
    fi

    case $num_pane in
    2)
        targets=(_ 1)
        directions=(_ h)
        percents=(_ '50%')
        ;;
    3)
        targets=(_ 1 2)
        directions=(_ h h)
        percents=(_ '66%' '50%')
        ;;
    4)
        targets=(_ 1 1 3)
        directions=(_ h v v)
        percents=(_ '50%' '50%' '50%')
        ;;
    5)
        targets=(_ 1 2 2 4)
        directions=(_ h h v v)
        percents=(_ '66%' '50%' '50%' '50%')
        ;;
    6)
        targets=(_ 1 2 1 3 5)
        directions=(_ h h v v v)
        percents=(_ '66%' '50%' '50%' '50%' '50%')
        ;;
    esac

    # start clients
    if [[ $window == 0 ]]; then
        mapfile -t clients_window_idx <<<"$(seq 1 $((num_pane - 1)))"
    else
        mapfile -t clients_window_idx <<<"$(seq 0 $((num_pane - 1)))"
    fi

    for i in "${clients_window_idx[@]}"; do
        echo "Starting client $client_id"
        if [[ $i == 0 ]]; then
            tmux_args=(new-window -n "group-${window}" -a -t "$session_name:group-$((window - 1))")
        else
            tmux_args=(split-window -"${directions[$i]}" -t "$session_name:group-${window}.${targets[$i]}" -l "${percents[$i]}")
        fi

        tmux "${tmux_args[@]}" \
            "$deffunc; settitle client-$client_id; \
            ssh -t ${hosts[$client_id]} \"$(gen_client_command "$client_id")\";"

        ((++client_id))
    done

    if [[ $window == 0 ]]; then
        num_clients_remain=$((num_clients_remain - num_pane + 1))
    else
        num_clients_remain=$((num_clients_remain - num_pane))
    fi
    ((++window))
done
# TODO: switch to a python script for better control/output
