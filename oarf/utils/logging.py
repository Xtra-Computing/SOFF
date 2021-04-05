import sys
import os
import logging
import threading
from oarf.utils.module import Module
from torch.utils.tensorboard import SummaryWriter


class TerminalFormatter(logging.Formatter):

    level_map = {
        logging.DEBUG: '[1m[94mDBG[0m',       # debug: blue
        logging.INFO: '[1m[92mINF[0m',        # info: green
        logging.WARNING: '[1m[93mWRN[0m',     # warning: yellow
        logging.ERROR: '[1m[91mERR[0m',       # error: red
        logging.CRITICAL: '[1m[95mCRT[0m',    # critical: purple
    }

    msg_color_map = {
        '->': '[2m',
        '<-': '[2m'
    }

    def format(self, record: logging.LogRecord):
        msg = record.getMessage()
        return "{} {}: {}{}[0m{}{}{}{}".format(
            self.level_map[record.levelno],
            self.formatTime(record, datefmt="%H:%M:%S"),
            (self.msg_color_map[msg[:2]] if len(msg) >= 2
                and msg[:2] in self.msg_color_map else ""),
            msg,
            os.linesep if record.stack_info else "",
            self.formatStack(record.stack_info) if record.stack_info else "",
            os.linesep if record.exc_info else "",
            self.formatException(record.exc_info) if record.exc_info else "")


class FileFormatter(logging.Formatter):
    level_map = {
        logging.DEBUG: 'DBG',        # debug: blue
        logging.INFO: 'INF',         # info: green
        logging.WARNING: 'WRN',      # warning: yellow
        logging.ERROR: 'ERR',        # error: red
        logging.CRITICAL: 'CRT',     # critical: purple
    }

    def format(self, record: logging.LogRecord):
        return "{} {} {} {}: {}{}{}{}{}".format(
            self.level_map[record.levelno],
            self.formatTime(record), record.name, record.thread,
            record.getMessage(), os.linesep if record.stack_info else "",
            self.formatStack(record.stack_info) if record.stack_info else "",
            os.linesep if record.exc_info else "",
            self.formatException(record.exc_info) if record.exc_info else "")


def init_logging(log_file=None, loglevel='info'):
    """
    A custom log that prints both to stderr and a file.
    This function only need to be called at the start of the main application.
    """

    loglevel = getattr(logging, loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    # add terminal handlers to the root logger
    log_handlers = []
    term_handler = logging.StreamHandler(sys.stdout)
    term_handler.setFormatter(TerminalFormatter())
    log_handlers.append(term_handler)

    # add file handlers to the root logger
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding='utf8')
        file_handler.setFormatter(FileFormatter())
        log_handlers.append(file_handler)

    # update root logger
    logging.basicConfig(level=loglevel, handlers=log_handlers)

    # handle all uncaught exceptions
    def uncaught_exception_handler(type, value, tb):
        logging.error("Uncaught exception.", exc_info=(type, value, tb))

    sys.excepthook = uncaught_exception_handler

    # Workaround for sys.excepthook thread bug
    # http://spyced.blogspot.com/2007/06/workaround-for-sysexcepthook-bug.html
    #
    # Call once from __main__ before creating any threads.
    # If using psyco, call psyco.cannotcompile(threading.Thread.run)
    # since this replaces a new-style class method.
    init_old = threading.Thread.__init__

    def init(self, *args, **kwargs):
        init_old(self, *args, **kwargs)
        run_old = self.run

        def run_with_except_hook(*args, **kw):
            try:
                run_old(*args, **kw)
            except (KeyboardInterrupt, SystemExit):
                sys.excepthook(*sys.exc_info())
                raise
            except Exception:
                sys.excepthook(*sys.exc_info())

        self.run = run_with_except_hook

    threading.Thread.__init__ = init


class Logger(Module):
    def __init__(self, logger_name, log_file, log_level,
                 tensorboard_log_dir, **kwargs):
        super().__init__(**kwargs)
        init_logging(log_file, log_level)
        self.log = logging.getLogger(logger_name)  # Path(__file__).stem)

        self.tfboard_writer = None if tensorboard_log_dir is None else \
            SummaryWriter(tensorboard_log_dir, flush_secs=30)
