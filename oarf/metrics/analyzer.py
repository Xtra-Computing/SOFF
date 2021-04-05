from __future__ import annotations  # for self-referencing typing
import time
import logging
import numpy as np
from typing import Callable, List, Union
from functools import partialmethod
from scipy.stats import wasserstein_distance, stats

log = logging.getLogger(__name__)


class Event:
    def __init__(
            self, parent_event: Union[Event, type(None)] = None,
            exit_hooks: List[Callable] = None):
        """
        exit_hooks: accept `self` as the only param, called when event ended
        """
        self.start_time = None
        self.end_time = None
        self.exit_hooks = exit_hooks

        self.sub_events = []
        self.parent_event = parent_event
        assert isinstance(self.parent_event, (Event, type(None)))

        # .append() is thread-safe, no need to worry about parallel events :)
        if self.parent_event is not None:
            self.parent_event.sub_events.append(self)

        self._required_records = ['start_time', 'end_time']

    def __enter__(self):
        self.start_time = time.time()
        log.info("-> {} @ {}".format(
            self.__class__.__name__, self.start_time))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        log.info("<- {} @ {}, total {} s".format(
            self.__class__.__name__, self.end_time,
            self.end_time - self.start_time))

        for r in self._required_records:
            assert getattr(self, r) is not None, \
                "{}.{} should not be none".format(self.__class__.__name__, r)

        if self.exit_hooks is not None:
            for hook in self.exit_hooks:
                hook(self)


class ServerTrainingEvent(Event):
    def __init__(self, *argc, **argv):
        super().__init__(*argc, **argv)
        self.data_emd = None
        self._required_records.extend(['data_emd'])

    def record(self, histos: List[List[int]]):
        # calculate emd and l1 distance
        self.EMD(histos)
        log.info("Dataset EMD: {}".format(self.data_emd))

    def EMD(self, histos: List[List[int]]):
        """return the sum of EMD of individual's dist and the global one"""
        global_hist = np.sum(histos, axis=0)
        global_dist = []
        for i, num in enumerate(global_hist):
            global_dist += [i] * num

        total_emd = 0
        for hist in histos:
            dist = []
            for i, num in enumerate(hist):
                dist += [i] * num
            total_emd += wasserstein_distance(dist, global_dist)

        self.data_emd = total_emd


class ClientTrainingEvent(Event):
    def __init__(self, *argc, **argv):
        super().__init__(*argc, **argv)
        self.data_skew = None
        self._required_records.extend(['data_skew'])

    def record(self, histo: List[int]):
        self.skew(histo)
        log.info("Dataset skew: {}".format(self.data_skew))

    def skew(self, histo: List[int]):
        dist = []
        for i, num in enumerate(histo):
            dist += [i] * num
        self.data_skew = stats.skew(dist)


class ServerCommRoundEvent(Event):
    def __init__(self, *argc, **argv):
        super().__init__(*argc, **argv)
        self.num_samples = None
        self.eval_loss = None
        self.additional_metrics = None
        self._required_records.extend(
            ['num_samples', 'eval_loss', 'eval_additional_metrics'])

    def record(self, num_samples, eval_loss, eval_additional_metrics):
        self.num_samples = num_samples
        self.eval_loss = eval_loss
        self.eval_additional_metrics = eval_additional_metrics


class ClientCommRoundEvent(Event):
    def __init__(self, *argc, **argv):
        super().__init__(*argc, **argv)
        self.num_samples = None
        self._required_records.extend(['num_samples'])

    def record(self, num_samples):
        self.num_samples = num_samples


class ForwardEvent(Event):
    def __init__(self, *argc, **argv):
        super().__init__(*argc, **argv)
        self.num_samples = None
        self._required_records.extend(['num_samples'])

    def record(self, num_samples):
        self.num_samples = num_samples


class BackwardEvent(Event):
    def __init__(self, *argc, **argv):
        super().__init__(*argc, **argv)
        self.num_samples = None
        self._required_records.extend(['num_samples'])

    def record(self, num_samples):
        self.num_samples = num_samples


class CompressionEvent(Event):
    pass


class DecompressionEvent(Event):
    pass


class EncryptionEvent(Event):
    pass


class DecryptionEvent(Event):
    pass


class SendDataEvent(Event):
    def __init__(self, *argc, **argv):
        super().__init__(*argc, **argv)
        self.bytes_sent = None
        self._required_records.extend(['bytes_sent'])

    def record(self, bytes_sent: int):
        self.bytes_sent = bytes_sent


class RecvDataEvent(Event):
    def __init__(self, *argc, **argv):
        super().__init__(*argc, **argv)
        self.bytes_received = None
        self._required_records.extend(['bytes_received'])

    def record(self, bytes_received: int):
        self.bytes_received = bytes_received


class BroadCastEvent(Event):
    def __init__(self, *argc, **argv):
        super().__init__(*argc, **argv)
        self.num_parties = None
        self._required_records.extend(['num_parties'])

    def record(self, num_parties: int):
        self.num_parties = num_parties


class AggregationEvent(Event):
    def __init__(self, *argc, **argv):
        super().__init__(*argc, **argv)
        self.num_parties = None
        self._required_records.extend(['num_parties'])

    def record(self, num_parties: int):
        self.num_parties = num_parties


class Analyzer:
    def __init__(self):
        self.root_event = None
        self.cur_event = None

    def training(self):
        return self.root_event

    def start_event(self, event_cls: type):
        """ start an event, and set cur_event """
        self.cur_event = event_cls(
            parent_event=self.cur_event, exit_hooks=[self._event_exit_hook])
        return self.cur_event

    def start_parallel_event(self, event_cls: type):
        """ start an event, but does not affect cur_event """
        return event_cls(parent_event=self.cur_event, exit_hooks=[])

    def _event_exit_hook(self, e: Event):
        log.debug("Event {} finished".format(e))
        # bubbling up along the tree until the last ongoing event
        while e is not None and e.end_time is not None:
            e = e.parent_event
        self.cur_event = e

    forward = partialmethod(start_event, ForwardEvent)
    backward = partialmethod(start_event, BackwardEvent)
    encrypt = partialmethod(start_event, EncryptionEvent)
    decrypt = partialmethod(start_event, DecryptionEvent)
    compress = partialmethod(start_event, CompressionEvent)
    decompress = partialmethod(start_event, DecompressionEvent)
    send_data = partialmethod(start_event, SendDataEvent)
    recv_data = partialmethod(start_event, RecvDataEvent)
    send_data_parallel = partialmethod(start_parallel_event, SendDataEvent)
    recv_data_parallel = partialmethod(start_parallel_event, RecvDataEvent)
    broadcast = partialmethod(start_event, BroadCastEvent)
    aggregate = partialmethod(start_event, AggregationEvent)

    def serailize(self):
        pass

    def deseralize(self):
        pass


class ServerAnalyzer(Analyzer):
    """
    analyzer class, events are organized in a tree structure
    """

    def __init__(self):
        super().__init__()
        self.root_event = ServerTrainingEvent()
        self.cur_event = self.root_event

    def start_event(self, event_cls: type):
        return super().start_event(event_cls)

    comm_round = partialmethod(start_event, ServerCommRoundEvent)

    def time_to_convergence(self):
        """calculate time to convergence"""
        pass

    def end_to_end_throughput(self):
        """calculate end-to-end throughput"""
        pass

    def time_decomposition(self):
        """time decomposition summary"""
        pass

    def __str__(self):
        pass


class ClientAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()
        self.root_event = ClientTrainingEvent()
        self.cur_event = self.root_event

    def start_event(self, event_cls: type):
        return super().start_event(event_cls)

    comm_round = partialmethod(start_event, ClientCommRoundEvent)

    def time_to_convergence(self):
        pass
