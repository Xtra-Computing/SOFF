"""Launch different server/clients"""

import os
import sys
import json
import argparse
import importlib
from os import listdir
from os.path import isfile, isdir
from pathlib import Path
from typing import Dict, List, Tuple, Type
from munch import Munch
from ...algorithms.base.base import Node
from ...utils.logging import init_logging
from ...utils.training import init_excepthook
from ...utils.arg_parser import BaseConfParser, r_hasattr
from ...utils.module import (
    camel_to_snake, load_all_direct_classes, load_all_submodules)

__module_root = Path(__file__).parent.parent.parent

alg_name_map: Dict[str, Type] = {
    f"{dir}.{camel_to_snake(name)}": cls
    # List algorithms
    for dir in listdir(f"{__module_root}/algorithms")
    if not dir == 'base'
    and isdir(f"{__module_root}/algorithms/{dir}")
    and isfile(f"{__module_root}/algorithms/{dir}/__init__.py")
    # List nodes for each algorithm
    for module in load_all_submodules(
        importlib.import_module(f"soff.algorithms.{dir}"))
    for name, cls in load_all_direct_classes(module).items()
    if issubclass(cls, Node) and not name.startswith('_')
}


class BaseLauncherConfParser(BaseConfParser):
    """Base class of launer config parser. Parse launcer config only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        launcher_args = self.add_argument_group(
            "Launcher Configs (S,C)")

        launcher_args.add_argument(
            '-l.a', '--launcher.algorithm',
            type=str, metavar='ALG', choices=alg_name_map.keys(),
            help="Server/Client algorithm class to instantiate. "
            "Use `-l.a <algorithm> -h` to view algorithm-specific help. "
            f"Supported algorithms: {list(alg_name_map.keys())}")
        launcher_args.add_argument(
            '-l.lf', '--launcher.log-file', type=str, metavar='FILE',
            help="Log file to save launcher arguments.")
        launcher_args.add_argument(
            '-l.ll', '--launcher.log-level', type=str,
            default='info', metavar='LVL',
            choices=['debug', 'info', 'warning', 'error', 'critical'],
            help="Set log levels")


        # debug arguments
        debug_args = self.add_argument_group(
            "Launcher Debug Configs (S,C)")

        debug_args.add_argument(
            '-dbg.pa', '--debug.pydevd-address',
            type=str, default=argparse.SUPPRESS, metavar='ADDR',
            help="Use remote debugging if the pydved's address is specified")
        debug_args.add_argument(
            '-dbg.pp', '--debug.pydevd-port',
            type=int, default=10802, metavar='PORT',
            help="specify the remote debuggin port")
        self.register_cfg_dep('--debug.pydevd-port', '--debug.pydevd-address')


class BaseLauncher:
    """Base launcher (for both server/clients)"""

    def __init__(self, cfg) -> None:
        # Enable debug if necessary
        if r_hasattr(cfg, 'debug.pydevd_address'):
            pydevd_pycharm = importlib.import_module('pydevd_pycharm')
            pydevd_pycharm.settrace(
                cfg.debug.pydevd_address, port=cfg.debug.pydevd_port,
                stdoutToServer=True, stderrToServer=True)

        self.algorithm = cfg.launcher.algorithm
        init_excepthook()

    @classmethod
    def parse_launcher_args(cls, parser_class=BaseLauncherConfParser) \
            -> Tuple[Munch, List]:
        """Parse basic launcher config"""
        conf_parser = parser_class(add_help=False)
        cfg, unknown = conf_parser.parse_known_args()

        # Algorithm must be specified. Otherwise print help and exit
        if cfg.launcher.algorithm is None:
            conf_parser.print_help()
            sys.exit(0)

        # Save arguments to log file
        print(json.dumps(cfg, indent=2))
        if cfg.launcher.log_file is not None:
            with open(cfg.launcher.log_file, 'w', encoding='utf-8') as file:
                file.write(f"({cls.__name__}) {' '.join(sys.argv[1:])}")
                file.write(2*os.linesep)
                file.write(json.dumps(cfg, indent=2))
                file.write(os.linesep)
        init_logging(cfg)

        return cfg, unknown
