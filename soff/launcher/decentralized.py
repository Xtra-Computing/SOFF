#!/bin/env python
"""Launcher script for decentralized algroithms"""

import ast
import sys
import math
import logging
from textwrap import dedent
from dataclasses import dataclass
from os import chdir, getcwd, makedirs, remove, system
from os.path import dirname, exists
from pathlib import Path
from typing import List, OrderedDict, Tuple
from libtmux.server import Server as TmuxServer
from libtmux.session import Session as TmuxSession
from .base import HWConfig, LauncherArgParser, TmuxPaneSplitter, set_title_func
from .node.base import alg_name_map
from ..algorithms.base.base_transceiver import DecentralizedBaseTransceiver


TMP_DIR = '/tmp/soff'
LOG_DIR = 'log'

log = logging.getLogger(__name__)


@dataclass
class HWConfigs:
    """Launcher hardware configs"""
    server: HWConfig
    edges: List[HWConfig]
    clients: List[HWConfig]


class Topology:
    def __init__(self) -> None:
        self.topology: OrderedDict[int, List[int]] = {}

    @staticmethod
    def from_args(args):
        def __check_topology(topology):
            # Basic type check
            assert isinstance(topology, OrderedDict)
            nodes = set(topology.keys())
            assert all(
                # Key type check
                isinstance(key, int)
                # Adj list type and value check
                and all(isinstance(adj, int) and adj in nodes for adj in val)
                # Prevent loopback connection
                and key not in val
                for key, val in topology.items())

        try:
            res = __class__()
            if args.topology.startswith('full'):
                num_nodes = int(args.topology.split(':')[-1])
                assert isinstance(num_nodes, int) and num_nodes >= 1
                res.topology = OrderedDict([
                    (i, [j for j in range(num_nodes) if j != i])
                    for i in range(num_nodes)])
            elif args.topology.startswith('ring'):
                num_nodes = int(args.topology.split(':')[-1])
                assert isinstance(num_nodes, int) and num_nodes >= 1
                res.topology = OrderedDict([
                    (i, [(i + num_nodes - 1) % num_nodes, (i + 1) % num_nodes])
                    for i in range(num_nodes)])
            else:
                res.topology = OrderedDict(ast.literal_eval(args.topology))
                # Complete bidirectional connection
                for key, val in res.topology.items():
                    for adj in val:
                        res.topology.setdefault(adj, [])
                        if key not in res.topology[adj]:
                            res.topology[adj] += [key]
                __check_topology(res.topology)
            return res
        except Exception as e:
            log.exception(
                "Invalid topology '%s'.", args.topology, exc_info=e)
            raise e

    @property
    def nodes(self):
        return self.topology.keys()


class DecentralizedLauncherArgParser(LauncherArgParser):
    def __init__(self) -> None:
        super().__init__()

        addr_parser = self.parser.add_mutually_exclusive_group()
        addr_parser.add_argument(
            '+s', '++socket-files', default=[], nargs='+',
            help="Socket files (for unix socket), must equals to the number of nodes")
        addr_parser.add_argument(
            '+t', '++tcp-addrs', default=[], nargs='+',
            help="TCP addresses (for TCP socket), must equals to the number of nodes")

        self.parser.add_argument(
            '+topo', '++topology', required=True, type=str,
            help="Decentralized topology, must be specified in the form of adj list. "
            "Topology assume dual-way connection. i.e. a -> b implies b -> a."
            "e.g.: '{1: [2, 3], 2: [1, 3], 3: [1, 2, 4], 4: [3,]}'. Two specical "
            "values full:<N> ring:<N> specifies full-connection and ring-connection "
            "of <N> nodes.")
        self.parser.add_argument(
            '+gc', '++group-nodes', type=int, default=1,
            help="Number of nodes per group (>1 to use multi-node launcher)")
        self.parser.add_argument(
            '+hw', '++hw-config-file', default='workers.txt',
            help="Hardware config file. Each line of the file should follow the "
            "format: '<client|server> <hostname> <gpu-idx>'")
        self.parser.add_argument(
            '+la', '++launcher-args', nargs='+', metavar='ARGS', default=[],
            help="Arguments for the node launcher")
        self.parser.add_argument(
            '+na', '++node-args', nargs='+', metavar='ARGS', default=[],
            help="Arguments shared by all edge aggregators")

    @property
    def available_algs(self):
        return set(
            name.split('.')[0] for name, cls in alg_name_map.items()
            if issubclass(cls, (DecentralizedBaseTransceiver)))


def parse_args():
    return DecentralizedLauncherArgParser().parser.parse_args()


def check_duplicate_session(args, tmux_server):
    """Check for duplicate sesisons"""
    if tmux_server.has_session(args.session_name):
        log.info("Session %s already started. Quitting...", args.session_name)
        sys.exit(1)


# Parse addresses .
def get_addr_options(args, topology) -> Tuple[str, OrderedDict[int, List[str]]]:
    """
    Get address options for both client and server
    Return: socket-type, server-addr-options, edge-addr-options, client-addr-options
    """

    add_options = OrderedDict()
    if args.tcp_addrs and len(args.tcp_addrs):
        socket_type = 'tcp'
        assert len(args.tcp_addrs) == len(topology.nodes)
        addr_options = OrderedDict((k, [
            '-dc/s.s', 'tcp', '-dc/s.a',
            f'0.0.0.0:{addr.split(":")[-1]}',
        ]) for k, addr in zip(topology.nodes, args.tcp_addrs))

        for k, v in topology.topology.items():
            if len(v) > 0:
                addr_options[k] += ['-dc/c.s', 'tcp', '-dc/c.as']
                addr_options[k] += [
                    f'{adj}::{args.tcp_addrs[topology.nodes.index(adj)]}'
                    for adj in v]
    else:
        socket_type = 'unix'
        args.socket_files = (
            args.socket_files or [
                f'{TMP_DIR}/fed-{args.session_name}-{i}.sock'
                for i in range(len(topology.nodes))])
        assert len(args.socket_files) == len(topology.nodes)

        addr_options = OrderedDict((k, [
            '-dc/s.s', 'unix', '-dc/s.a', addr,
        ]) for k, addr in zip(topology.nodes, args.socket_files))
        for k, v in topology.topology.items():
            if len(v) > 0:
                addr_options[k] += ['-dc/c.s', 'unix', '-dc/c.as']
                addr_options[k] += [
                    f'{adj}::{args.socket_files[list(topology.nodes).index(adj)]}'
                    for adj in v]

        for socket_file in args.socket_files:
            makedirs(dirname(socket_file), 0o700, exist_ok=True)

    return socket_type, addr_options


def read_hw_configs(args, topology) -> HWConfigs:
    """Read hardware configs"""
    with open(args.hw_config_file, 'r', encoding='utf8') as file:
        lines = file.readlines()
    result = HWConfigs(HWConfig('localhost', 0), [], [])
    for line in lines:
        node_type, hostname, gpuindex = line.split()
        if node_type == 'edge':
            result.edges.append(HWConfig(hostname, int(gpuindex)))
        else:
            log.warning("Ignore unrecognized hw config: %s", line)

    assert len(result.edges) >= len(topology.nodes), (
        "Insufficient lines of worker configs for nodes")

    assert all(len(set(
        node.hostname for node in result.edges[
            launcher_id * args.group_nodes:
            (launcher_id + 1) * args.group_nodes
        ])) == 1 for launcher_id in
        range(math.ceil(len(topology.nodes) / args.group_nodes))
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


def start_nodes(
        tmux_server: TmuxServer, args, hwconfs: HWConfigs,
        node_alg_name: str, addr_options: OrderedDict[int, List[str]],
        topology, lock_file: str) -> TmuxSession:

    def launcher_script_path(launcher_id):
        return f'{TMP_DIR}/{args.session_name}_edge_launcher{launcher_id}.sh'

    def gen_per_node_args(node_id: int):

        # Generate execution plan for special connection types
        execution_plan = ''
        if node_alg_name == 'd_fedavg.transceiver':
            if args.topology.startswith('full'):
                plan = (((node_id,), (n for n in topology.nodes if n !=
                        node_id)), ((n for n in topology.nodes), ()))
                execution_plan = f'-davg.ep "{plan}"'
            elif args.topology.startswith('ring'):
                incoming_id = (
                    node_id + len(topology.nodes) - 1) % len(topology.nodes)
                outgoing_id = (node_id + 1) % len(topology.nodes)
                plan = ((((node_id,), (outgoing_id,)),) +
                        tuple(((incoming_id, node_id), (outgoing_id,))
                              for _ in range(len(topology.nodes) - 2)) +
                        (((incoming_id, node_id), ()),))
                execution_plan = f'-davg.ep "{plan}"'

        # Calculate incoming edges
        num_endpoints = sum(
            (1 if node_id in v else 0) for k, v in topology.topology.items())

        log_args = (rf"""
                    -lg.df "{LOG_DIR}/{args.log_dir}/edge{node_id}.csv" \
                    -lg.tld "{LOG_DIR}/{args.log_dir}/tfboard"
                """).strip()

        return (rf"""
                    -tc.id {node_id} \
                    {execution_plan} \
                    {' '.join(addr_options[node_id])} \
                    {' '.join(args.node_args)} \
                    -ss.n {num_endpoints} \
                    -hw.gs {hwconfs.edges[node_id].gpuindex} \
                    {log_args} {'-dgb.st' if args.skip_training else ''}
                """).strip()

    launcher_id = 0

    num_nodes_left = len(topology.nodes)
    for launcher_id in range(((len(topology.nodes) - 1) // args.group_nodes) + 1):
        launcher_class = 'node' if args.group_nodes == 1 else 'multi_node'
        with open(launcher_script_path(launcher_id), 'w', encoding='utf8') as f:
            f.write(dedent(rf"""
                #!/bin/bash
                cd {getcwd()}
                {DecentralizedLauncherArgParser.exports(args)}
                python3 \
                    -um soff.launcher.node.{launcher_class} \
                    -l.a {node_alg_name} \
                    -dt.fs.n "{len(topology.nodes)}" \
                    -l.lf "{LOG_DIR}/{args.log_dir}/launcher_node_{launcher_id}.log" \
                    {' ' if args.group_nodes == 1 else f'-l.mn.n {args.group_nodes}' } \
                    {' '.join(args.launcher_args)} \
                    {
                        gen_per_node_args(list(topology.nodes)[launcher_id * args.group_nodes])
                        if args.group_nodes == 1 else
                        " ".join(
                            " -- " + gen_per_node_args(topology.nodes[launcher_id * args.group_nodes + i])
                            for i in range(min(num_nodes_left, args.group_nodes)))
                    } \
                    2>"{LOG_DIR}/{args.log_dir}/launcher_node_{launcher_id}.err"
                """).strip())
            num_nodes_left -= args.group_nodes

        # Send script to host
        hostname = hwconfs.edges[launcher_id * args.group_nodes].hostname
        system(
            f"scp {launcher_script_path(launcher_id)} "
            f"{hostname}:{launcher_script_path(launcher_id)}")

        launcher_id += 1

    launcher_id = 0
    hang_tmux = 'sleep 1h' if args.tmux_debug else ':'

    # Start a session, ssh to server, and start the first node
    session = tmux_server.new_session(
        args.session_name, attach=False, window_name='server',
        window_command=(
            f"bash -c '{set_title_func};"
            f"settitle launcher-n-0; "
            f"ssh -t {hwconfs.edges[0].hostname} bash {launcher_script_path(0)}; "
            f"echo first_node_done > {lock_file}; "
            f"{hang_tmux};'"))

    session.set_option('pane-border-status', 'top', True)
    session.set_option('pane-border-format', '#T', True)

    pane_splitter = TmuxPaneSplitter(session, len(topology.nodes))

    for launcher_id in list(range(((len(topology.nodes) - 1) // args.group_nodes) + 1))[1:]:
        hostname = hwconfs.edges[launcher_id * args.group_nodes].hostname
        cmd = dedent(
            f"bash -c '{set_title_func}; "
            f"settitle launcher-n-{launcher_id}; "
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
    tmux_server = TmuxServer()

    # Parse hierarchy
    topology = Topology.from_args(args)

    # Generate/read configs
    check_duplicate_session(args, tmux_server)
    _, addr_options = get_addr_options(args, topology)
    hw_configs = read_hw_configs(args, topology)
    log_metadata(args)

    # Cleanup files of the previous run
    lock_file = f"{TMP_DIR}/{args.session_name}.lock"
    if exists(lock_file):
        remove(lock_file)

    # Start server
    node_alg_name = list(filter(
        lambda item: item[0].startswith(args.algorithm) and
        issubclass(item[1], DecentralizedBaseTransceiver), alg_name_map.items()))[0][0]
    log.info("Using algorithm: %s", node_alg_name)
    start_nodes(
        tmux_server, args, hw_configs,
        node_alg_name, addr_options, topology, lock_file)


if __name__ == "__main__":
    main()
