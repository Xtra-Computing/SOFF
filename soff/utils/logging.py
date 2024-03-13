"""Custom logging configs"""

import os
import csv
import sys
import time
import logging
import pathlib
import threading
from typing import List, Optional
from munch import Munch
from torch import Tensor
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import SummaryWriter
from .arg_parser import BaseConfParser, r_hasattr


class LogConfParser(BaseConfParser):
    """Parse configs for logging facilities"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        log_args = self.add_argument_group(
            "Logging Configs (S,C)")

        log_args.add_argument(
            '-lg.df', '--log.data-file', type=str, metavar='FILE',
            help="Path to the data csv file")
        log_args.add_argument(
            '-lg.tld', '--log.tensorboard-log-dir', type=str, metavar='DIR',
            help="Tensorflow log directory")
        log_args.add_argument(
            '-lg.tfs', '--log.tensorboard-flush-secs',
            default=30, type=int, metavar='SECS',
            help="Tensorflow fluch interval")
        self.register_cfg_dep(
            '--log.tensorboard-flush-secs', '--log.tensorboard-log-dir')


class TerminalFormatter(logging.Formatter):
    """Formatter for terminal output"""
    level_map = {
        logging.DEBUG: '\x1B[1m\x1B[94mDBG\x1B[0m',       # debug: blue
        logging.INFO: '\x1B[1m\x1B[92mINF\x1B[0m',        # info: green
        logging.WARNING: '\x1B[1m\x1B[93mWRN\x1B[0m',     # warning: yellow
        logging.ERROR: '\x1B[1m\x1B[91mERR\x1B[0m',       # error: red
        logging.CRITICAL: '\x1B[1m\x1B[95mCRT\x1B[0m',    # critical: purple
    }

    msg_color_map = {
        '->': '\x1B[2m',
        '<-': '\x1B[2m'
    }

    def format(self, record: logging.LogRecord):
        msg = record.getMessage()
        msg_color = self.msg_color_map[msg[:2]] \
            if len(msg) >= 2 and msg[:2] in self.msg_color_map else ''
        result = f"{self.level_map[record.levelno]} "\
            f"{self.formatTime(record, datefmt='%H:%M:%S')}: " \
            f"{msg_color}{msg}\x1B[0m"
        if record.stack_info:
            result += f"{os.linesep}{self.formatStack(record.stack_info)}"
        if record.exc_info:
            result += f"{os.linesep}{self.formatException(record.exc_info)}"
        return result


class FileFormatter(logging.Formatter):
    """Formatter for file output"""
    level_map = {
        logging.DEBUG: 'DBG',
        logging.INFO: 'INF',
        logging.WARNING: 'WRN',
        logging.ERROR: 'ERR',
        logging.CRITICAL: 'CRT',
    }

    def format(self, record: logging.LogRecord):
        result = f"{self.level_map[record.levelno]} {self.formatTime(record)} "\
            f"{record.name} {record.thread}: {record.getMessage()}"
        if record.stack_info:
            result += f"{os.linesep}{self.formatStack(record.stack_info)}"
        if record.exc_info:
            result += f"{os.linesep}{self.formatException(record.exc_info)}"
        return result


def init_logging(cfg: Munch):
    """
    A custom log that prints both to stderr and a file.
    This function only need to be called at the start of the main application.
    """
    file = cfg.launcher.log_file if r_hasattr(cfg, 'launcher.log_file') else None
    level = cfg.launcher.log_level if r_hasattr(cfg, 'launcher.log_level') else 'INFO'

    loglevel = getattr(logging, level.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError(f"Invalid log level: {loglevel}")

    # add terminal handlers to the root logger
    log_handlers = []
    term_handler = logging.StreamHandler(sys.stdout)
    term_handler.setFormatter(TerminalFormatter())
    log_handlers.append(term_handler)

    # add file handlers to the root logger
    if file is not None:
        pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file, encoding='utf8')
        file_handler.setFormatter(FileFormatter())
        log_handlers.append(file_handler)

    # update root logger
    logging.basicConfig(level=loglevel, handlers=log_handlers)


class DataLogger:
    """
    Log to both file and tensorboard. If tag_suffix is provided,
    "/<tag_suffix>" will be appended to all log entry tags
    """

    def __init__(self, cfg: Munch, tag_suffix: Optional[str] = None) -> None:
        if (r_hasattr(cfg, 'log.tensorboard_log_dir') and
                cfg.log.tensorboard_log_dir is not None):
            pathlib.Path(cfg.log.tensorboard_log_dir).mkdir(
                parents=True, exist_ok=True)

        self.tfboard_writer = (SummaryWriter(
            log_dir=cfg.log.tensorboard_log_dir,
            flush_secs=cfg.log.tensorboard_flush_secs
        ) if r_hasattr(cfg, 'log.tensorboard_log_dir')
            and cfg.log.tensorboard_log_dir is not None else None)

        if r_hasattr(cfg, 'log.data_file') and cfg.log.data_file is not None:
            pathlib.Path(os.path.dirname(cfg.log.data_file)).mkdir(
                parents=True, exist_ok=True)

        self.csv_file = (open(
            cfg.log.data_file, 'a', encoding='utf-8', newline=''
        ) if r_hasattr(cfg, 'log.data_file')
            and cfg.log.data_file is not None else None)

        self.csv_writer = (csv.writer(
            self.csv_file, delimiter=','
        ) if self.csv_file is not None else None)
        self.csv_writer_lock = threading.Lock()

        if self.csv_writer is not None:
            self.csv_writer.writerow(["timestamp", "step", "tag", "value"])

        self.tag_suffix = f"/{tag_suffix}" if tag_suffix is not None else ""

    def add_scalar(self, tag, scalar_value, global_step=None, **kwargs):
        """Log scalar value to tfboard and file"""
        if self.tfboard_writer is not None:
            self.tfboard_writer.add_scalar(
                f"{tag}{self.tag_suffix}", scalar_value, global_step,
                **kwargs)
        if self.csv_writer:
            with self.csv_writer_lock:
                self.csv_writer.writerow([
                    time.time(), global_step,
                    f"{tag}{self.tag_suffix}", scalar_value])

    def add_scalars(
            self, main_tag, tag_scalar_dict,
            global_step=None, walltime=None, **kwargs):
        """Log scalar values to tfboard and file"""
        if self.tfboard_writer is not None:
            self.tfboard_writer.add_scalars(main_tag, tag_scalar_dict={
                f"{tag}{self.tag_suffix}": val for tag,
                val in tag_scalar_dict.items()
            }, global_step=global_step, walltime=walltime, **kwargs)
        if self.csv_writer:
            for tag, val in tag_scalar_dict.items():
                self.csv_writer.writerow([
                    time.time(), global_step,
                    f"{main_tag}/{tag}{self.tag_suffix}", val])

    def add_histogram(self, tag, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_histogram"""
        if self.tfboard_writer:
            self.tfboard_writer.add_histogram(
                f"{tag}{self.tag_suffix}", *args, **kwargs)

    def add_histogram_raw(self, tag, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_histogram_raw"""
        if self.tfboard_writer:
            self.tfboard_writer.add_histogram_raw(
                f"{tag}{self.tag_suffix}", *args, **kwargs)

    def add_image(self, tag, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_image"""
        if self.tfboard_writer:
            self.tfboard_writer.add_image(
                f"{tag}{self.tag_suffix}", *args, **kwargs)

    def add_images(self, tag, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_images"""
        if self.tfboard_writer:
            self.tfboard_writer.add_images(
                f"{tag}{self.tag_suffix}", *args, **kwargs)

    def add_image_with_boxes(self, tag, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_image_with_boxes"""
        if self.tfboard_writer:
            self.tfboard_writer.add_image_with_boxes(
                f"{tag}{self.tag_suffix}", *args, **kwargs)

    def add_figure(self, tag, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_figure"""
        if self.tfboard_writer:
            self.tfboard_writer.add_figure(
                f"{tag}{self.tag_suffix}", *args, **kwargs)

    def add_video(self, tag, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_video"""
        if self.tfboard_writer:
            self.tfboard_writer.add_video(
                f"{tag}{self.tag_suffix}", *args, **kwargs)

    def add_audio(self, tag, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_audio"""
        if self.tfboard_writer:
            self.tfboard_writer.add_audio(
                f"{tag}{self.tag_suffix}", *args, **kwargs)

    def add_text(
            self, tag, text_string, global_step=None, **kwargs):
        """Log text to tfboard and file"""
        if self.tfboard_writer:
            self.tfboard_writer.add_text(
                f"{tag}{self.tag_suffix}",
                text_string, global_step=None, **kwargs)
        if self.csv_writer:
            self.csv_writer.writerow([
                time.time(), global_step,
                f"{tag}{self.tag_suffix}", text_string])

    def add_onnx_graph(self, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_onnx_graph"""
        if self.tfboard_writer:
            self.tfboard_writer.add_onnx_graph(*args, **kwargs)

    def add_graph(self, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_graph"""
        if self.tfboard_writer:
            self.tfboard_writer.add_graph(*args, **kwargs)

    def add_embedding(self, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_embedding"""
        if self.tfboard_writer:
            self.tfboard_writer.add_embedding(*args, **kwargs)

    def add_pr_curve(self, tag, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_pr_curve"""
        if self.tfboard_writer:
            self.tfboard_writer.add_pr_curve(
                f"{tag}{self.tag_suffix}", *args, **kwargs)

    def add_pr_curve_raw(self, tag, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_pr_curve_raw"""
        if self.tfboard_writer:
            self.tfboard_writer.add_pr_curve_raw(
                f"{tag}{self.tag_suffix}", *args, **kwargs)

    def add_custom_scalars_multilinechart(
            self, tags: List[str], *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_custom_scalars_multilinechart"""
        if self.tfboard_writer:
            self.tfboard_writer.add_custom_scalars_multilinechart(
                [f"{tag}{self.tag_suffix}" for tag in tags], *args, **kwargs)

    def add_custom_scalars_marginchart(self, tags, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_custom_scalars_marginchart"""
        if self.tfboard_writer:
            self.tfboard_writer.add_custom_scalars_marginchart(
                [f"{tag}{self.tag_suffix}" for tag in tags], *args, **kwargs)

    def add_mesh(self, tag, *args, **kwargs):
        """Wrapper of self.tfboard_writer.add_mesh"""
        if self.tfboard_writer:
            self.tfboard_writer.add_mesh(
                f"{tag}{self.tag_suffix}", *args, **kwargs)

    def register_hparams(self, cfg: Munch, metrics: Optional[List[str]] = None):
        """
        Registeres hyperparameters and metrics (read from `cfg`).
        Subsquent metric updates with `add_scalar` usign keys specified in
        `metrics` automatically populate metric reuslts in hyperparameters.
        """
        if self.tfboard_writer is not None:
            if metrics is None:
                metrics = []
            # Register hyper parameters from cfg, and metrics from args
            exp, ssi, sei = hparams({
                key: (
                    val if isinstance(val, (int, float, str, bool, Tensor))
                    else str(val)
                ) for key, val in BaseConfParser.flatten(cfg)
            }, metric_dict={
                f"{metric}{self.tag_suffix}": None for metric in metrics
            })
            self.tfboard_writer._get_file_writer().add_summary(exp)
            self.tfboard_writer._get_file_writer().add_summary(ssi)
            self.tfboard_writer._get_file_writer().add_summary(sei)

    def flush(self):
        """Flush tensorboard and binary log file"""
        if self.tfboard_writer:
            self.tfboard_writer.flush()
        if self.csv_file:
            self.csv_file.flush()

    def close(self):
        """Close tensorboard and binary log file"""
        if self.tfboard_writer:
            self.tfboard_writer.close()
        if self.csv_file:
            self.csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        self.close()
