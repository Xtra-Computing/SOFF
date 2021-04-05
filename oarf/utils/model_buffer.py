import torch
import threading
import logging
from enum import Enum, unique
from oarf.utils.training import all_params

log = logging.getLogger(__name__)


@unique
class _ModelState(Enum):
    IDLE = 0
    RECEIVING = 1
    AGGREGATING = 2
    READY = 3


class ClientModel:
    def __init__(self, net_template: torch.nn.Module):
        self.id = None
        self.lock = threading.Lock()
        # state-machine: idle -> receiving -> ready -> aggregating -> idle.
        self.state = _ModelState.IDLE
        self.grad = [
            torch.zeros(size=param.shape)
            for param in all_params(net_template)
        ]

    def set_grad(self, grads):
        """set gradient for all network parameters (including buffers)"""
        assert len(grads) == len(self.grad)
        for s_grad, grad in zip(self.grad, grads):
            assert s_grad.shape == grad.shape
            s_grad.set_(grad.type(s_grad.dtype))


class ModelBuffer(list):
    """
    A buffer to temporarily store models received from clients, with lifecycle
    managements and multithreading support.
    """

    def __init__(self, n: int, net_template: torch.nn.Module):
        self.lock = threading.Lock()
        # represents the number of idle nets
        self.sem_nets_idle = threading.Semaphore(n)
        # represents the number of ready nets
        self.sem_nets_ready = threading.Semaphore(0)

        super().__init__()
        for i in range(n):
            self.append(ClientModel(net_template))

    def get_net_for_receive(self):
        # select a vacant slot
        self.sem_nets_idle.acquire()
        self.lock.acquire()
        net = None
        idx = None
        for i, net_selected in enumerate(self):
            if net_selected.state == _ModelState.IDLE:
                net_selected.state = _ModelState.RECEIVING
                idx = i
                net = net_selected
                break
        self.lock.release()

        if idx is not None:
            return idx, net
        else:
            return None

    def release_as_ready(self, idx):
        self.lock.acquire()
        if self[idx].state == _ModelState.RECEIVING:
            self[idx].state = _ModelState.READY
        self.lock.release()
        self.sem_nets_ready.release()

    def get_net_for_aggregate(self):
        self.sem_nets_ready.acquire()
        self.lock.acquire()
        net = None
        idx = None
        for i, net_selected in enumerate(self):
            if net_selected.state == _ModelState.READY:
                net_selected.state = _ModelState.AGGREGATING
                idx = i
                net = net_selected
                break
        self.lock.release()

        if idx is not None:
            return idx, net
        else:
            return None

    def release_as_idle(self, idx):
        self.lock.acquire()
        if self[idx].state == _ModelState.AGGREGATING:
            self[idx].state = _ModelState.IDLE
        self.lock.release()
        self.sem_nets_idle.release()
