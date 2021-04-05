import time
import logging

log = logging.getLogger(__name__)


def timeit(method):
    """A timer decorator"""
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        log.info("{}: {} ms".format(method.__name__, (te - ts) * 1000))
        return result

    return timed
