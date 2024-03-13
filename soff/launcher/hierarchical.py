#!/bin/env python
"""Launcher script for hierarchical algroithms"""

import ast
import sys
import math
import logging
from time import sleep
from textwrap import dedent
from dataclasses import dataclass
from os.path import dirname, exists, isfile
from os import chdir, getcwd, makedirs, remove, system
from pathlib import Path
from typing import Callable, List, Tuple, Union
from libtmux.server import Server as TmuxServer
from libtmux.session import Session as TmuxSession
from .base import HWConfig, LauncherArgParser, TmuxPaneSplitter, set_title_func
from .node.base import alg_name_map
from ..algorithms.base.base_server import BaseServer
from ..algorithms.base.base_client import BaseClient
from ..algorithms.base.base_transceiver import HierarchicalBaseTransceiver


TMP_DIR = '/tmp/soff'
LOG_DIR = 'log'

log = logging.getLogger(__name__)


@dataclass
class HWConfigs:
    """Launcher hardware configs"""
    server: HWConfig
    edges: List[HWConfig]
    clients: List[HWConfig]


class Hierarchy:
    def __init__(self) -> None:
        self.hierarchy = None

    @staticmethod
    def from_args(args):
        def __check_hierarchy(group):
            assert isinstance(group, (int, tuple)), \
                f"Hierarcy must be a (nested) tuple of ints. got {group}."
            if isinstance(group, tuple):
                for elem in group:
                    __check_hierarchy(elem)

        try:
            res = __class__()
            res.hierarchy = ast.literal_eval(args.hierarchy)
            __check_hierarchy(res.hierarchy)
            return res
        except Exception as e:
            log.exception("invalid hierarchy '%s'.",
                          args.hierarchy, exc_info=e)
            sys.exit(1)

    @property
    def num_edge_aggregators(self):
        """Nmber of edge aggregators"""
        def __get_sum(group):
            return 1 if isinstance(group, int) else \
                1 + sum(__get_sum(elem) for elem in group)
        return __get_sum(self.hierarchy) - 1

    @property
    def num_clients(self):
        """Nmber of edge clients (leafs)"""
        def __get_sum(group):
            return group if isinstance(group, int) else \
                sum(__get_sum(elem) for elem in group)
        return __get_sum(self.hierarchy)

    def pre_order_dfs(self, visit: Callable[[Union[Tuple, int], int], None]):
        """
        visit: a callable that accepts:
            arg1: tuple or int, specifying the architecture
            arg2: current traverse level, starting from 0
        """
        def __pre_order_dfs(group, visit, level: int):
            visit(group, level)
            if isinstance(group, tuple):
                for elem in group:
                    __pre_order_dfs(elem, visit, level + 1)
        __pre_order_dfs(self.hierarchy, visit, 0)


class HierarchicalLauncherArgParser(LauncherArgParser):
    def __init__(self) -> None:
        super().__init__()
        addr_parser = self.parser.add_mutually_exclusive_group()
        addr_parser.add_argument(
            '+s', '++socket-files', default=[], nargs='+',
            help="Socket files (for unix socket), must equals to the number of "
            "1 (root node) + intermediate node")
        addr_parser.add_argument(
            '+t', '++tcp-addrs', default=[], nargs='+',
            help="TCP addresses (for TCP socket), must equals to the number of "
            "1 (root node) + intermediate node. Intermediate node addresses are "
            "assigned in the same order as pre-order DFS traverse.")

        self.parser.add_argument(
            '+hi', '++hierarchy', type=str, required=True,
            help="node/leaf node hierarchy '()' for edge aggregator, and numbers "
            "for leaf nodes connected to an edge aggregator. For example, "
            "'3' means 3 clients. '(3, 3)' means 2 intermediate node and 3 leaf "
            "node for each intermediate node. More complex hierarchy like "
            "'(2, 2, (4, 4))' is also possible (total 5 intermediate node and 12 "
            "clients).")
        self.parser.add_argument(
            '+ge', '++group-edge-aggregators', type=int, default=1,
            help="Number of edge nodes per group (>1 to use multi-node launcher)")
        self.parser.add_argument(
            '+gc', '++group-clients', type=int, default=1,
            help="Number of clients per group (>1 to use multi-node launcher)")
        self.parser.add_argument(
            '+hw', '++hw-config-file', default='workers.txt',
            help="Hardware config file. Each line of the file should follow the "
            "format: '<client|server> <hostname> <gpu-idx>'")
        self.parser.add_argument(
            '+sa', '++server-args', nargs='+', metavar='ARGS', default=[],
            help="Arguments for the server")
        self.parser.add_argument(
            '+ela', '++edge-launcher-args', nargs='+', metavar='ARGS', default=[],
            help="Arguments for the edge aggregator launcher")
        self.parser.add_argument(
            '+ea', '++edge-args', nargs='+', metavar='ARGS', default=[],
            help="Arguments shared by all edge aggregators")
        self.parser.add_argument(
            '+cla', '++client-launcher-args', nargs='+', metavar='ARGS', default=[],
            help="Arguments for the client launcher")
        self.parser.add_argument(
            '+ca', '++client-args', nargs='+', metavar='ARGS', default=[],
            help="Arguments shared by all clients")

    @property
    def available_algs(self):
        return set(
            name.split('.')[0] for name, cls in alg_name_map.items()
            if issubclass(cls, (BaseServer, BaseClient)))


def parse_args():
    return HierarchicalLauncherArgParser().parser.parse_args()


def check_duplicate_session(args, tmux_server):
    """Check for duplicate sesisons"""
    if tmux_server.has_session(args.session_name):
        log.info("Session %s already started. Quitting...", args.session_name)
        sys.exit(1)


# Parse addresses .
def get_addr_options(args, hierarchy) -> Tuple[str, List[str], List[List[str]], List[List[str]]]:
    """
    Get address options for both client and server
    Return: socket-type, server-addr-options, edge-addr-options, client-addr-options
    """

    edge_aggregator_addr_options = []
    client_addr_options = []
    if args.tcp_addrs and len(args.tcp_addrs):
        socket_type = 'tcp'
        assert len(args.tcp_addrs) == hierarchy.num_edge_aggregators + 1

        server_addr_options = [
            '-dc.s', 'tcp', '-dc.a', f'0.0.0.0:{args.tcp_addrs[0].split(":")[-1]}']
        cur_addr_idx = 0
        addr_idx_stack = [0]

        def assign_addr_options(group, level):
            nonlocal cur_addr_idx, addr_idx_stack
            nonlocal edge_aggregator_addr_options, client_addr_options
            if level > 0:
                edge_aggregator_addr_options.append([
                    '-dc/s.s', 'tcp', '-dc/s.a', f'0.0.0.0:{args.tcp_addrs[cur_addr_idx+1].split(":")[-1]}',
                    '-dc/c.s', 'tcp', '-dc/c.a', f'{args.tcp_addrs[addr_idx_stack[level-1]]}'])
                cur_addr_idx += 1
                addr_idx_stack = addr_idx_stack[:level]
                addr_idx_stack.append(cur_addr_idx)
            if isinstance(group, int):
                for _ in range(group):
                    client_addr_options.append([
                        '-dc.s', 'tcp', '-dc.a', args.tcp_addrs[addr_idx_stack[level]]])
        hierarchy.pre_order_dfs(assign_addr_options)
    else:
        socket_type = 'unix'
        args.socket_files = (
            args.socket_files or [
                f'{TMP_DIR}/fed-{args.session_name}-{i}.sock'
                for i in range(hierarchy.num_edge_aggregators + 1)
            ])
        assert len(args.socket_files) == hierarchy.num_edge_aggregators + 1

        server_addr_options = ['-dc.s', 'unix', '-dc.a', args.socket_files[0]]
        cur_addr_idx = 0
        addr_idx_stack = [0]

        def assign_addr_options(group, level):
            nonlocal addr_idx_stack, cur_addr_idx
            nonlocal edge_aggregator_addr_options, client_addr_options
            if level > 0:
                edge_aggregator_addr_options.append([
                    '-dc/s.s', 'unix', '-dc/s.a', args.socket_files[cur_addr_idx+1],
                    '-dc/c.s', 'unix', '-dc/c.a', args.socket_files[addr_idx_stack[level-1]]])
                cur_addr_idx += 1
                addr_idx_stack = addr_idx_stack[:level]
                addr_idx_stack.append(cur_addr_idx)
            if isinstance(group, int):
                for _ in range(group):
                    client_addr_options.append([
                        '-dc.s', 'unix', '-dc.a', args.socket_files[addr_idx_stack[level]]])
        hierarchy.pre_order_dfs(assign_addr_options)
        for socket_file in args.socket_files:
            makedirs(dirname(socket_file), 0o700, exist_ok=True)
    return (
        socket_type, server_addr_options,
        edge_aggregator_addr_options, client_addr_options)


def read_hw_configs(args, hierarchy: Hierarchy) -> HWConfigs:
    """Read hardware configs"""
    with open(args.hw_config_file, 'r', encoding='utf8') as file:
        lines = file.readlines()
    result = HWConfigs(HWConfig('localhost', 0), [], [])
    for line in lines:
        node_type, hostname, gpuindex = line.split()
        if node_type == 'server':
            result.server = HWConfig(hostname, int(gpuindex))
        elif node_type == 'edge':
            result.edges.append(HWConfig(hostname, int(gpuindex)))
        elif node_type == 'client':
            result.clients.append(HWConfig(hostname, int(gpuindex)))
        else:
            log.warning("Ignore unrecognized hw config: %s", line)

    assert len(result.edges) >= hierarchy.num_edge_aggregators, (
        "Insufficient lines of worker configs for edge aggregators")
    assert len(result.clients) >= hierarchy.num_clients, (
        "Insufficient lines of worker configs for clients")

    assert all(len(set(
        cli.hostname for cli in result.clients[
            launcher_id * args.group_clients:
            (launcher_id + 1) * args.group_clients
        ])) == 1 for launcher_id in
        range(math.ceil(hierarchy.num_clients / args.group_clients))
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
        hierarchy: Hierarchy, lock_file: str) -> TmuxSession:
    """Start the server's algorithm"""
    log_dir = args.log_dir
    log_options = (rf"""
                -lg.df "{LOG_DIR}/{log_dir}/server.csv" \
                -lg.tld "{LOG_DIR}/{log_dir}/tfboard"
            """).strip() if not args.no_log else ''

    num_endpoints = (
        len(hierarchy.hierarchy) if isinstance(hierarchy.hierarchy, tuple)
        else hierarchy.hierarchy)
    launcher_script_path = f'{TMP_DIR}/{args.session_name}_server_launcher.sh'
    with open(launcher_script_path, 'w', encoding='utf8') as file:
        file.write(dedent(rf"""
            #!/bin/bash
            cd {getcwd()}
            {HierarchicalLauncherArgParser.exports(args)}
            python3 \
                -um soff.launcher.node.node \
                -l.a {server_alg_name} \
                -l.lf "{LOG_DIR}/{log_dir}/launcher_server.log" \
                -ss.n "{num_endpoints}" \
                -dt.fs.n "{hierarchy.num_clients}" \
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
        while not Path(args.socket_files[0]).is_socket():
            sleepcount = wait_or_kill(sleepcount)
    elif socket_type == 'tcp':
        while not system(
                f"nc -zv {args.tcp_addrs[0].split(':')[0]} "
                f"{args.tcp_addrs[0].split(':')[-1]} >/dev/null 2>&1"):
            sleepcount = wait_or_kill(sleepcount)
    sleep(1)


def start_edge_aggregators(
        pane_splitter: TmuxPaneSplitter, hierarchy: Hierarchy, args,
        hwconfs: HWConfigs, aggregator_alg_name: str,
        edge_aggregator_addr_options: List[List[str]]):

    def launcher_script_path(launcher_id):
        return f'{TMP_DIR}/{args.session_name}_edge_launcher{launcher_id}.sh'

    num_clients_counted = 0

    def gen_per_edge_aggregator_args(edge_id: int, group):
        nonlocal num_clients_counted
        num_endpoints = (
            group if isinstance(group, int)
            else sum(1 if isinstance(e, tuple) else e for e in group))

        # Count num clients connected to leaf nodes
        base_id = num_clients_counted if isinstance(group, int) else 0
        num_clients_counted += num_endpoints

        log_args = (rf"""
                    -lg.df "{LOG_DIR}/{args.log_dir}/edge{edge_id}.csv" \
                    -lg.tld "{LOG_DIR}/{args.log_dir}/tfboard"
                """).strip()

        return (rf"""
                    {' '.join(edge_aggregator_addr_options[edge_id])} \
                    {' '.join(args.edge_args)} \
                    -ss.n {num_endpoints} \
                    -ss.bi {base_id} \
                    -hw.gs {hwconfs.edges[edge_id].gpuindex} \
                    {log_args} {'-dgb.st' if args.skip_training else ''}
                """).strip()

    group_edges = args.group_edge_aggregators
    launcher_id = 0
    num_edges_left = hierarchy.num_edge_aggregators

    def __gen_edge_aggregator_launcher(group, level):
        nonlocal launcher_id, num_edges_left
        if level == 0:
            return

        launcher_class = 'node' if group_edges == 1 else 'multi_node'
        with open(launcher_script_path(launcher_id), 'w', encoding='utf8') as f:
            f.write(dedent(rf"""
                #!/bin/bash
                cd {getcwd()}
                {HierarchicalLauncherArgParser.exports(args)}
                python3 \
                    -um soff.launcher.node.{launcher_class} \
                    -l.a {aggregator_alg_name} \
                    -dt.fs.n "{hierarchy.num_clients}" \
                    -l.lf "{LOG_DIR}/{args.log_dir}/launcher_edge_{launcher_id}.log" \
                    {' ' if group_edges == 1 else f'-l.mn.n {group_edges}' } \
                    {' '.join(args.edge_launcher_args)} \
                    {
                        gen_per_edge_aggregator_args(launcher_id * group_edges, group)
                        if group_edges == 1 else
                        " ".join(
                            " -- " + gen_per_edge_aggregator_args(launcher_id * group_edges + i, group)
                            for i in range(min(num_edges_left, group_edges)))
                    } \
                    2>"{LOG_DIR}/{args.log_dir}/launcher_edge_{launcher_id}.err"
                """).strip())
            num_edges_left -= group_edges

        # Send script to host
        hostname = hwconfs.edges[launcher_id * group_edges].hostname
        system(
            f"scp {launcher_script_path(launcher_id)} "
            f"{hostname}:{launcher_script_path(launcher_id)}")

        launcher_id += 1
    hierarchy.pre_order_dfs(__gen_edge_aggregator_launcher)

    launcher_id = 0
    hang_tmux = 'sleep 1h' if args.tmux_debug else ':'

    def __start_edge_aggregator(group, level):
        nonlocal launcher_id
        if level == 0:
            return
        hostname = hwconfs.edges[launcher_id * group_edges].hostname
        cmd = dedent(
            f"bash -c '{set_title_func}; "
            f"settitle launcher-e-{launcher_id}; "
            f"ssh -t {hostname} bash {launcher_script_path(launcher_id)}; "
            f"{hang_tmux};'")
        pane_splitter.start_new_pane(cmd)
        launcher_id += 1
    hierarchy.pre_order_dfs(__start_edge_aggregator)


def start_clients(
        pane_splitter: TmuxPaneSplitter, hierarchy: Hierarchy, args,
        hwconfs: HWConfigs, client_alg_name: str, client_addr_options: List[List[str]]):
    """Start all clients"""

    def launcher_script_path(launcher_id):
        return f'{TMP_DIR}/{args.session_name}_cli_launcher{launcher_id}.sh'

    def gen_per_client_args(cli_id: int):
        log_args = (rf"""
                    -lg.df "{LOG_DIR}/{args.log_dir}/client{cli_id}.csv" \
                    -lg.tld "{LOG_DIR}/{args.log_dir}/tfboard"
                """).strip()

        return (rf"""
                    {' '.join(client_addr_options[cli_id])} \
                    {' '.join(args.client_args)} \
                    -hw.gs {hwconfs.clients[cli_id].gpuindex} \
                    {log_args} {'-dgb.st' if args.skip_training else ''}
                """).strip()

    # Generate per-launcher config
    launcher_id = 0
    group_clients = args.group_clients
    num_clients_left = hierarchy.num_clients

    def __gen_clients_launcher(group, _):
        nonlocal launcher_id, num_clients_left
        if not isinstance(group, int):
            return
        for _ in range(group):
            launcher_class = 'node' if group_clients == 1 else 'multi_node'
            with open(launcher_script_path(launcher_id), 'w', encoding='utf8') as f:
                f.write(dedent(rf"""
                    #!/bin/bash
                    cd {getcwd()}
                    {HierarchicalLauncherArgParser.exports(args)}
                    python3 \
                        -um soff.launcher.node.{launcher_class} \
                        -l.a {client_alg_name} \
                        -l.lf "{LOG_DIR}/{args.log_dir}/launcher_cli_{launcher_id}.log" \
                        {' ' if group_clients == 1 else f'-l.mn.n {group_clients}' } \
                        {' '.join(args.client_launcher_args)} \
                        {
                            gen_per_client_args(launcher_id * group_clients)
                            if group_clients == 1 else
                            " ".join(
                                " -- " + gen_per_client_args(launcher_id * group_clients + i)
                                for i in range(min(num_clients_left, group_clients)))
                        } \
                        2>"{LOG_DIR}/{args.log_dir}/launcher_cli_{launcher_id}.err"
                    """).strip())
                num_clients_left -= args.group_clients

            # Send script to host
            hostname = hwconfs.clients[launcher_id * group_clients].hostname
            system(
                f"scp {launcher_script_path(launcher_id)} "
                f"{hostname}:{launcher_script_path(launcher_id)}")
            launcher_id += 1

    hierarchy.pre_order_dfs(__gen_clients_launcher)

    # Start all client launchers
    launcher_id = 0
    hang_tmux = 'sleep 1h' if args.tmux_debug else ':'

    def __start_client(group, _):
        nonlocal launcher_id
        if not isinstance(group, int):
            return
        for _ in range(group):
            hostname = hwconfs.clients[launcher_id * group_clients].hostname
            cmd = dedent(
                f"bash -c '{set_title_func}; "
                f"settitle launcher-c-{launcher_id}; "
                f"ssh -t {hostname} bash {launcher_script_path(launcher_id)}; "
                f"{hang_tmux};'")
            pane_splitter.start_new_pane(cmd)
            launcher_id += 1
    hierarchy.pre_order_dfs(__start_client)


def main():
    """Start everything"""

    # Change to project root dir
    chdir(Path(__file__).absolute().parent.parent.parent)

    # Create runtime directory
    Path.mkdir(Path(TMP_DIR), mode=0o700, parents=False, exist_ok=True)

    # Parse args and init tmux server
    args = parse_args()
    tmux_server = TmuxServer()

    # Parse hierarchy
    hierarchy = Hierarchy.from_args(args)

    # Generate/read configs
    check_duplicate_session(args, tmux_server)
    (socket_type, serv_addr_options,
     edge_aggregator_addr_options,
     cli_addr_options) = get_addr_options(args, hierarchy)
    hw_configs = read_hw_configs(args, hierarchy)
    log_metadata(args)

    # Cleanup files of the previous run
    lock_file = f"{TMP_DIR}/{args.session_name}.lock"
    if exists(lock_file):
        remove(lock_file)

    # Start server
    server_alg_name = list(filter(
        lambda item: item[0].startswith(args.algorithm) and
        issubclass(item[1], BaseServer), alg_name_map.items()))[0][0]
    log.info("Using server alg: %s", server_alg_name)
    tmux_session = start_server(
        tmux_server, args, hw_configs,
        server_alg_name, serv_addr_options, hierarchy, lock_file)
    wait_for_server_start(args, socket_type, lock_file)

    pane_splitter = TmuxPaneSplitter(
        tmux_session, hierarchy.num_edge_aggregators+hierarchy.num_clients)

    # Start edge aggregators
    edge_aggregator_alg_name = list(filter(
        lambda item: item[0].startswith(args.algorithm) and
        issubclass(item[1], HierarchicalBaseTransceiver), alg_name_map.items()))[0][0]
    start_edge_aggregators(
        pane_splitter, hierarchy, args, hw_configs,
        edge_aggregator_alg_name, edge_aggregator_addr_options)

    # Start clients
    client_alg_name = list(filter(
        lambda item: item[0].startswith(args.algorithm) and
        issubclass(item[1], BaseClient), alg_name_map.items()))[0][0]
    log.info("Using client alg: %s", client_alg_name)
    start_clients(
        pane_splitter, hierarchy, args, hw_configs, client_alg_name, cli_addr_options)


if __name__ == "__main__":
    main()
