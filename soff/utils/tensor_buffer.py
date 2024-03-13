"""A simple buffer for temporarily storing models/gradients"""
from abc import abstractmethod
from functools import partial
from queue import Queue
import threading
import logging
from enum import Enum, unique
from typing import Any, Callable

log = logging.getLogger(__name__)


@unique
class _BufferState(Enum):
    IDLE = 0
    RECEIVING = 1
    AGGREGATING = 2
    READY = 3


class BufferSlot:
    """
    Buffer slot follows a state-machine:
    idle -> receiving -> ready -> aggregating -> idle.
    """

    def __init__(self):
        self.client_id = None
        self.lock = threading.Lock()
        self.state = _BufferState.IDLE
        self.data: Any = None

        self.release_as_idle: Callable[[], None]
        """Handler to be installed by TensorBuffer"""
        self.release_as_ready: Callable[[], None]
        """Handler to be installed by TensorBuffer"""

    def set_data(self, data):
        """Set buffer slot data"""
        self.data = data

    def set_id(self, client_id):
        """Set client_id"""
        self.client_id = client_id


class _TensorBuffer:
    @abstractmethod
    def get_slot_for_receive(self) -> BufferSlot:
        raise NotImplementedError

    @abstractmethod
    def get_slot_for_aggregate(self) -> BufferSlot:
        raise NotImplementedError


class TensorBuffer(list, _TensorBuffer):
    """
    A buffer to temporarily store models received from clients, with lifecycle
    managements and multithreading support.
    """

    def __init__(self, n: int):
        self.num_clients = n

        self.lock = threading.Lock()
        # represents the number of idle nets
        self.sem_nets_idle = threading.Semaphore(n)
        # represents the number of ready nets
        self.sem_nets_ready = threading.Semaphore(0)

        super().__init__()
        for _ in range(self.num_clients):
            self.append(BufferSlot())

    def get_slot_for_receive(self) -> BufferSlot:
        """Select a vacant slot"""
        self.sem_nets_idle.acquire()
        with self.lock:
            slot = None
            idx = None
            for i, net_selected in enumerate(self):
                if net_selected.state == _BufferState.IDLE:
                    net_selected.state = _BufferState.RECEIVING
                    idx = i
                    slot = net_selected
                    break

        assert idx is not None and slot is not None
        self.__install_handler(slot, idx)
        return slot

    def __release_as_ready(self, idx) -> None:
        """Release a slot as ready after receiving"""
        with self.lock:
            if self[idx].state == _BufferState.RECEIVING:
                self[idx].state = _BufferState.READY
        self.sem_nets_ready.release()

    def get_slot_for_aggregate(self) -> BufferSlot:
        """Get a filled slot for aggregate"""
        self.sem_nets_ready.acquire()
        with self.lock:
            slot = None
            idx = None
            for i, net_selected in enumerate(self):
                if net_selected.state == _BufferState.READY:
                    net_selected.state = _BufferState.AGGREGATING
                    idx = i
                    slot = net_selected
                    break

        assert idx is not None and slot is not None
        self.__install_handler(slot, idx)
        return slot

    def __release_as_idle(self, idx) -> None:
        """Release a slot as idle (ready for receiving) after aggregation"""
        with self.lock:
            if self[idx].state == _BufferState.AGGREGATING:
                self[idx].state = _BufferState.IDLE

        self.sem_nets_idle.release()

    def __install_handler(self, buffer: BufferSlot, idx: int):
        buffer.release_as_idle = partial(self.__release_as_idle, idx)
        buffer.release_as_ready = partial(self.__release_as_ready, idx)


class QueuedTensorBuffer(Queue, _TensorBuffer):
    """
    A buffer to temporarily store models received from clients, with lifecycle
    managements and multithreading support. (Using FIFO strategy)
    """

    def __init__(self, n: int) -> None:
        super().__init__(n)

    def __install_handler(self, buffer: BufferSlot):
        buffer.release_as_idle = lambda: None
        buffer.release_as_ready = lambda: None

    def get_slot_for_receive(self) -> BufferSlot:
        """Select a vacant slot"""
        slot = BufferSlot()
        self.put(slot)

        self.__install_handler(slot)
        return slot

    def get_slot_for_aggregate(self) -> BufferSlot:
        """Get a filled slot for aggregate"""
        slot = self.get()
        self.__install_handler(slot)
        return slot
