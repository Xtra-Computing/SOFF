#!/bin/env python
"""Launcher script for client-server algroithms"""

import re
import sys
import math
import logging
from time import sleep
from textwrap import dedent
from dataclasses import dataclass
from os.path import dirname, exists, isfile
from os import chdir, getcwd, makedirs, remove, system
from pathlib import Path
from typing import List, Tuple
from libtmux.server import Server as TmuxServer
from libtmux.session import Session as TmuxSession
from .base import HWConfig, LauncherArgParser, TmuxPaneSplitter, set_title_func, LOG_DIR, TMP_DIR
from .node.base import alg_name_map
from ..algorithms.base.base_server import ClientServerBaseServer
from ..algorithms.base.base_client import ClientServerBaseClient


log = logging.getLogger(__name__)


@dataclass
class HWConfigs:
    """Launcher hardware configs"""
    server: HWConfig
    clients: List[HWConfig]


class ClientServerLauncherArgParser(LauncherArgParser):
    def __init__(self) -> None:
        super().__init__()

        addr_parser = self.parser.add_mutually_exclusive_group()
        addr_parser.add_argument(
            '+s', '++socket-file', default='',
            help="Socket file (for unix socket)")
        addr_parser.add_argument(
            '+t', '++tcp-addr', default='',
            help="TCP address (for TCP socket)")
        self.parser.add_argument(
            '+n', '++num-clients', type=int, required=True,
            help='Number of clinets to launch')
        self.parser.add_argument(
            '+g', '++group-clients', type=int, default=1,
            help="Number of clients per group (>1 to use multi-node launcher)")
        self.parser.add_argument(
            '+hw', '++hw-config-file', default='workers.txt',
            help="Hardware config file. Each line of the file should follow the "
            "format: '<client|server> <hostname> <gpu-idx>'")
        self.parser.add_argument(
            '+sa', '++server-args', nargs='+', metavar='ARGS', default=[],
            help="Arguments for the server")
        self.parser.add_argument(
            '+cla', '++client-launcher-args', nargs='+', metavar='ARGS', default=[],
            help="Arguments for the client launcher")
        self.parser.add_argument(
            '+ca', '++client-args', nargs='+', metavar='ARGS', default=[],
            help="Arguments shared by all clients")
        self.parser.add_argument(
            '+pca', '++per-client-args', nargs='+', metavar='ARGS', default=[],
            help="Arguments per client. Use -<N> prefix to specify args for client <N>."
            "e.g. '-0 -a -b -3 -c -d' will assign '-a -b' to client 0 and '-c -d' to client 3.")

    @property
    def available_algs(self):
        return set(
            name.split('.')[0] for name, cls in alg_name_map.items()
            if issubclass(cls, (ClientServerBaseServer, ClientServerBaseClient)))


def parse_args():
    return ClientServerLauncherArgParser().parser.parse_args()


def check_duplicate_session(args, tmux_server):
    """Check for duplicate sesisons"""
    if tmux_server.has_session(args.session_name):
        log.info("Session %s already started. Quitting...", args.session_name)
        sys.exit(1)


def get_addr_options(args) -> Tuple[str, List[str], List[str]]:
    """
    Get address options for both client and server
    Return: socket-type, server-addr-options, client-addr-options
    """
    if args.tcp_addr:
        socket_type = 'tcp'
        server_addr_options = [
            '-dc.s', 'tcp', '-dc.a', f'0.0.0.0:{args.tcp_addr.split(":")[-1]}']
        client_addr_options = [
            '-dc.s', 'tcp', '-dc.a', args.tcp_addr]
    else:
        socket_type = 'unix'
        args.socket_file = (
            args.socket_file or f'{TMP_DIR}/fed-{args.session_name}.sock')
        server_addr_options = ['-dc.s', 'unix', '-dc.a', args.socket_file]
        client_addr_options = ['-dc.s', 'unix', '-dc.a', args.socket_file]
        makedirs(dirname(args.socket_file), 0o700, exist_ok=True)

    return socket_type, server_addr_options, client_addr_options


def read_hw_configs(args) -> HWConfigs:
    """Read hardware configs"""
    with open(args.hw_config_file, 'r', encoding='utf8') as file:
        lines = file.readlines()
    result = HWConfigs(HWConfig('localhost', 0), [])
    for line in lines:
        node_type, hostname, gpuindex = line.split()
        if node_type == 'server':
            result.server = HWConfig(hostname, int(gpuindex))
        elif node_type == 'client':
            result.clients.append(HWConfig(hostname, int(gpuindex)))
        else:
            log.warning("Ignore unrecognized hw config: %s", line)

    assert len(result.clients) >= args.num_clients, \
        "Insufficient lines of worker configs for clients"

    assert all(len(set(
        cli.hostname for cli in result.clients[
            launcher_id * args.group_clients:
            (launcher_id + 1) * args.group_clients
        ])) == 1 for launcher_id in
        range(math.ceil(args.num_clients / args.group_clients))
    ), "Clients in the same launcher must use the same host"

    return result


def log_metadata(args):
    """Log metadata to log dir"""
    if args.no_log:
        return

    log.info("Creating log directory ...")
    log_dir = args.log_dir
    makedirs(f'{LOG_DIR}/{log_dir}', exist_ok=True)

    # Log invoke command
    with open(
            f'{LOG_DIR}/{log_dir}/invoke_command.txt', 'w',
            encoding='utf8') as file:
        file.write(f"{' '.join(sys.argv)}\n{args.comment}\n")

    # Log current code status
    if system('git rev-parse --is-inside-work-tree >/dev/null 2>&1') == 0:
        system(f'git rev-parse HEAD > {LOG_DIR}/{log_dir}/git_revision.txt')
        system(f'git diff --patch > {LOG_DIR}/{log_dir}/git_patch.txt')
    else:
        if exists('git_status/git_revision.txt'):
            system(
                'cp git_status/git_revision.txt '
                f'{LOG_DIR}/{log_dir}/git_revision.txt')
        if exists('git_status/git_patch.patch'):
            system(
                f'cp git_status/git_patch.patch '
                f'{LOG_DIR}/{log_dir}/git_patch.txt')


def start_server(
        tmux_server: TmuxServer, args, hwconfs: HWConfigs,
        server_alg_name: str, server_addr_options: List[str],
        lock_file: str) -> TmuxSession:
    """Start the server's algorithm"""
    log_dir = args.log_dir
    log_options = (rf"""
                -lg.df "{LOG_DIR}/{log_dir}/server.csv" \
                -lg.tld "{LOG_DIR}/{log_dir}/tfboard"
            """).strip() if not args.no_log else ''
    launcher_script_path = f'{TMP_DIR}/{args.session_name}_server_launcher.sh'
    with open(launcher_script_path, 'w', encoding='utf8') as file:
        file.write(dedent(rf"""
            #!/bin/bash
            cd {getcwd()}
            {ClientServerLauncherArgParser.exports(args)}
            python3 \
                -um soff.launcher.node.node \
                -l.a {server_alg_name} \
                -l.lf "{LOG_DIR}/{log_dir}/launcher_server.log" \
                -cs.n "{args.num_clients}" \
                -ss.n "{args.num_clients}" \
                -dt.fs.n "{args.num_clients}" \
                -hw.gs "{hwconfs.server.gpuindex}" \
                {' '.join(server_addr_options)} \
                {' '.join(args.server_args)} \
                {log_options} \
                2>"{LOG_DIR}/{log_dir}/server.err"
            """).strip())

    # Send the generated script to server
    log.info("Sending server launch script")
    system(
        f"scp {launcher_script_path} "
        f"{hwconfs.server.hostname}:{launcher_script_path}")

    log.info("Starting server ...")

    hang_tmux = 'sleep 1h' if args.tmux_debug else ':'

    # Start a session, ssh to server, and execute the script
    session = tmux_server.new_session(
        args.session_name, attach=False, window_name='server',
        window_command=(
            f"bash -c '{set_title_func};"
            f"settitle server; "
            f"ssh -t {hwconfs.server.hostname} bash {launcher_script_path}; "
            f"echo server_done > {lock_file}; "
            f"{hang_tmux};'"))

    session.set_option('pane-border-status', 'top', True)
    session.set_option('pane-border-format', '#T', True)
    return session


def wait_for_server_start(args, socket_type, lock_file):
    """Wait for the server to start"""
    def wait_or_kill(sleepcount):
        # Check for server failure
        if isfile(lock_file):
            with open(lock_file, 'r', encoding='utf8') as file:
                line = file.readline().strip()
            if line == 'server_done':
                log.error("Server Exited (probably due to error)")
                sys.exit(1)

        sleep(1)
        if sleepcount > 300:
            with open(lock_file, 'w', encoding='utf8') as file:
                file.write('wait_timeout')
            sys.exit(1)
        return sleepcount + 1

    sleepcount = 0
    if socket_type == 'unix':
        while not Path(args.socket_file).is_socket():
            sleepcount = wait_or_kill(sleepcount)
    elif socket_type == 'tcp':
        while not system(
                f"nc -zv {args.tcp_addr.split(':')[0]} "
                f"{args.tcp_addr.split(':')[-1]} >/dev/null 2>&1"):
            sleepcount = wait_or_kill(sleepcount)
    sleep(1)


def start_clients(
        pane_splitter: TmuxPaneSplitter, args, hwconfs: HWConfigs,
        client_alg_name: str, client_addr_options: List[str]):
    """Start all clients"""

    def launcher_script_path(launcher_id):
        return f'{TMP_DIR}/{args.session_name}_cli_launcher{launcher_id}.sh'

    # Split per-client args
    per_client_args = dict()
    cur_client_id = -1
    for arg in args.per_client_args:
        match = re.match(r'^\-(\d+)$', arg)
        if match:
            cur_client_id = int(match[1])
            per_client_args[cur_client_id] = []
        else:
            per_client_args[cur_client_id].append(arg)

    def gen_per_client_args(cli_id: int):
        log_args = (rf"""
                    -lg.df "{LOG_DIR}/{args.log_dir}/client{cli_id}.csv" \
                    {' '.join(per_client_args[cli_id]) if cli_id in per_client_args else ''} \
                    -lg.tld "{LOG_DIR}/{args.log_dir}/tfboard"
                """).strip()

        return (rf"""
                    {' '.join(client_addr_options)} \
                    {' '.join(args.client_args)} \
                    -hw.gs {hwconfs.clients[cli_id].gpuindex} \
                    {log_args} {'-dgb.st' if args.skip_training else ''}
                """).strip()

    # Generate per-launcher config
    group_clients = args.group_clients
    num_launchers = math.ceil(args.num_clients / group_clients)
    num_clients_left = args.num_clients
    for launcher_id in range(num_launchers):
        launcher_class = 'node' if group_clients == 1 else 'multi_node'
        with open(launcher_script_path(launcher_id), 'w') as f:
            f.write(dedent(rf"""
                #!/bin/bash
                cd {getcwd()}
                {ClientServerLauncherArgParser.exports(args)}
                python3 \
                    -um soff.launcher.node.{launcher_class} \
                    -l.a {client_alg_name} \
                    -l.lf "{LOG_DIR}/{args.log_dir}/launcher{launcher_id}.log" \
                    {' ' if group_clients == 1 else f'-l.mn.n {group_clients}' } \
                    {' '.join(args.client_launcher_args)} \
                    {
                        gen_per_client_args(launcher_id * group_clients)
                        if group_clients == 1 else
                        " ".join(
                            " -- " + gen_per_client_args(launcher_id * group_clients + i)
                            for i in range(min(num_clients_left, group_clients)))
                    } \
                    2>"{LOG_DIR}/{args.log_dir}/launcher{launcher_id}.err"
                """).strip())
            num_clients_left -= args.group_clients

        # Send script to host
        hostname = hwconfs.clients[launcher_id * group_clients].hostname
        system(
            f"scp {launcher_script_path(launcher_id)} "
            f"{hostname}:{launcher_script_path(launcher_id)}")

    # Start all client launchers
    hang_tmux = 'sleep 1h' if args.tmux_debug else ':'
    for launcher_id in range(num_launchers):
        hostname = hwconfs.clients[launcher_id * group_clients].hostname
        cmd = dedent(
            f"bash -c '{set_title_func}; "
            f"settitle launcher-{launcher_id}; "
            f"ssh -t {hostname} bash {launcher_script_path(launcher_id)}; "
            f"{hang_tmux};'")
        pane_splitter.start_new_pane(cmd)


def main():
    """Start everything"""

    # Change to project root dir
    chdir(Path(__file__).absolute().parent.parent.parent)

    # Create runtime directory
    Path.mkdir(Path(TMP_DIR), mode=0o700, parents=False, exist_ok=True)

    # Parse args and init tmux server
    args = parse_args()
    print(args)
    tmux_server = TmuxServer()

    # Generate/read configs
    check_duplicate_session(args, tmux_server)
    socket_type, serv_addr_options, cli_addr_options = get_addr_options(args)
    hw_configs = read_hw_configs(args)
    log_metadata(args)

    # Cleanup files of the previous run
    lock_file = f"{TMP_DIR}/{args.session_name}.lock"
    if exists(lock_file):
        remove(lock_file)

    # Start server
    server_alg_name = list(filter(
        lambda item: item[0].startswith(args.algorithm) and
        issubclass(item[1], ClientServerBaseServer), alg_name_map.items()))[0][0]
    log.info("Using server alg: %s", server_alg_name)
    tmux_session = start_server(
        tmux_server, args, hw_configs,
        server_alg_name, serv_addr_options, lock_file)
    wait_for_server_start(args, socket_type, lock_file)

    num_launchers = math.ceil(args.num_clients / args.group_clients)
    pane_splitter = TmuxPaneSplitter(tmux_session, num_launchers)

    # Start clients
    client_alg_name = list(filter(
        lambda item: item[0].startswith(args.algorithm) and
        issubclass(item[1], ClientServerBaseClient), alg_name_map.items()))[0][0]
    log.info("Using client alg: %s", client_alg_name)
    start_clients(
        pane_splitter, args, hw_configs, client_alg_name, cli_addr_options)


if __name__ == "__main__":
    main()
