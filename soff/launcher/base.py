from datetime import datetime
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from textwrap import dedent
from typing import Dict, List, Optional
from libtmux.session import Session as TmuxSession

LOG_DIR = 'log'
TMP_DIR = '/tmp/soff'
log = logging.getLogger(__name__)


@dataclass
class HWConfig:
    """Node hardware config"""
    hostname: str
    gpuindex: int


set_title_func = dedent(r"""
    settitle() {
        export PS1="\[\e[32m\]\u@\h \[\e[33m\]\w\[\e[0m\]\n$ "
        echo -ne "\e]0; $1 \a"
    }
    """).strip()


class TmuxPaneSplitter:
    """Tmux pane split manager"""

    @dataclass
    class SplitCfg:
        target: List[int]
        vertical: List[bool]
        percent: List[int]

    split_cfgs: Dict[int, SplitCfg] = {
        2: SplitCfg(
            target=[0],
            vertical=[False],
            percent=[50],
        ), 3: SplitCfg(
            target=[0, 1],
            vertical=[False, False],
            percent=[66, 50],
        ), 4: SplitCfg(
            target=[0, 0, 1],
            vertical=[False, True, True],
            percent=[50, 50, 50],
        ), 5: SplitCfg(
            target=[0, 1, 1, 2],
            vertical=[False, False, True, True],
            percent=[66, 50, 50, 50],
        ), 6: SplitCfg(
            target=[0, 1, 0, 1, 2],
            vertical=[False, False, True, True, True],
            percent=[66, 50, 50, 50, 50],
        ),
    }

    def __init__(self, tmux_session: TmuxSession, total_num_panes: int) -> None:
        """
        tmux_session: the existing session to split
        total_num_panes: number of panes to add
        """
        # Invariants
        self.tmux_session = tmux_session
        self.total_num_panes = total_num_panes

        # State vars
        self.cur_launcher_id = 0
        self.cur_window = tmux_session.windows[0]
        self.cur_win_panes = [self.cur_window.panes[0].pane_id]
        self.split_cfg: Optional[self.__class__.SplitCfg] = self.split_cfgs[
            min(total_num_panes + 1, 5)]

    def start_new_pane(self, cmd):
        """start a new pane with cmd"""

        if self.cur_launcher_id >= self.total_num_panes:
            log.error("Luncher id exceeds originally planned.")
            return

        if (len(self.cur_window.panes)
                + int(self.cur_window == self.tmux_session.windows[0])) % 6 == 0:
            num_launchers_remain = self.total_num_panes - self.cur_launcher_id
            self.split_cfg = self.split_cfgs[min(num_launchers_remain, 6)]
            self.cur_window = self.tmux_session.new_window(
                window_name=f"group-{len(self.tmux_session.windows)}",
                attach=False,
                window_shell=cmd)
            self.cur_win_panes = [self.cur_window.panes[0].pane_id]
        else:
            assert self.split_cfg is not None
            idx = len(self.cur_window.panes) - 1
            new_pane = self.cur_window.split_window(
                target=self.cur_win_panes[self.split_cfg.target[idx]],
                vertical=self.split_cfg.vertical[idx],
                percent=self.split_cfg.percent[idx], attach=False,
                shell=cmd)
            self.cur_win_panes.append(new_pane.pane_id)

        self.cur_launcher_id += 1


class LauncherArgParser:
    def __init__(self) -> None:
        self.parser = ArgumentParser(
            prefix_chars='+', formatter_class=ArgumentDefaultsHelpFormatter)
        self.parser.add_argument(
            '+a', '++algorithm', type=str, metavar='ALG', default='fedavg',
            choices=self.available_algs,
            help=f"Algorithm to use. Available: {list(self.available_algs)}")
        self.parser.add_argument(
            '+sn', '++session-name', type=str, default='comm',
            help="Tmux session name")
        self.parser.add_argument(
            '+envs', '++environment-vars', type=str, nargs='+', default=[],
            help="Environment variables shared by all nodes "
            "(e.g. PYTHONFAULTHANDLER=1 CUDA_LAUNCH_BLOCKING=1 TORCH_CUDNN_V8_API_DISABLED=1)")

        self.parser.add_argument(
            '+l', '++log-dir', metavar='NAME',
            default=datetime.now().strftime('%F_%H-%M-%S'),
            help=f"Log directory name. logs will be stored under {LOG_DIR}/<NAME>")
        self.parser.add_argument(
            '+nl', '++no-log', action='store_true',
            help="Do not enable log")

        self.parser.add_argument(
            '+C', '++comment', type=str, default='',
            help='Comment to add to invoke_command.txt')

        self.parser.add_argument(
            '+st', '++skip-training', action='store_true',
            help="Debug option to skip training")
        self.parser.add_argument(
            '+tdbg', '++tmux-debug', action='store_true',
            help="Debug option to debug tmux")

    @property
    def available_algs(self):
        raise NotImplementedError

    @staticmethod
    def exports(args):
        return '\n'.join([f"export {var}" for var in args.environment_vars])
