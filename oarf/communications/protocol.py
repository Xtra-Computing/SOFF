import json
import socket
import threading
import numpy as np
import logging
from enum import Enum, unique
from oarf.metrics.analyzer import SendDataEvent, RecvDataEvent
from oarf.compressors.compress_utils import (
    Offset, pack_int, pack_raw_data, unpack_int, unpack_raw_data)

# This file defines a custom socket comm protocol
log = logging.getLogger(__name__)


@unique
class MessageType(Enum):
    HANDSHAKE = b'h'
    TRAINING_CONFIG = b't'
    MODEL = b'm'
    GRADIENT = b'g'
    WEIGHTED_SUM_GRADIENT = b'w'  # for fednova, \(p_id_i^{(t-1)}\)
    SYNC_INFOS = b's'             # for fedavg-like algorithm, metadatas
    CLIENT_INFOS = b'c'
    STEP_WEIGHTS = b'S'           # for fednova, \(a_i in the paper\)
    RSA_PUBKEY = b'r'
    GRAD_SPLIT = b'f'             # forward message to another client
    BYE = b'B'                    # shutdown signal, usually from srv to cli
    _BYE = b'b'                   # for dispatcher's internal use


class Protocol:
    meta_lock = threading.Lock()
    send_locks = {}
    recv_locks = {}
    """ensure that only 1 thread sends/recvs from one socket"""

    @staticmethod
    def send_data(socket: socket.socket, msg_type: MessageType,
                  data: bytearray, analyzer: SendDataEvent = None):
        assert (len(msg_type.value) == 1)

        header = bytearray()

        header.extend(msg_type.value)
        header.extend(np.uint64([len(data)]).tobytes())

        with Protocol.meta_lock:
            if socket not in Protocol.send_locks:
                Protocol.send_locks[socket] = threading.Lock()

        with Protocol.send_locks[socket]:
            log.debug("Sending data ->{}: {} (len={})".format(
                socket.fileno(), msg_type, len(data)))
            socket.sendall(header)
            socket.sendall(data)

        if analyzer is not None:
            analyzer.record(bytes_sent=len(data))
        log.debug("Data sent")

    @staticmethod
    def recv_data(socket: socket.socket, analyzer: RecvDataEvent = None):
        log.debug("receiving data")

        data = bytearray()
        piece = bytearray()
        with Protocol.meta_lock:
            if socket not in Protocol.recv_locks:
                Protocol.recv_locks[socket] = threading.Lock()

        with Protocol.recv_locks[socket]:
            while len(piece) < 9:
                try:
                    subpiece = socket.recv(9 - len(piece))
                except ConnectionResetError:
                    if analyzer is not None:
                        analyzer.record(bytes_received=0)
                    return (MessageType._BYE, bytearray())

                if len(subpiece) == 0:
                    if analyzer is not None:
                        analyzer.record(bytes_received=0)
                    return (MessageType._BYE, bytearray())
                piece.extend(subpiece)

            msg_type = MessageType(bytes(piece[0:1]))
            length = int(np.frombuffer(piece[1:9], dtype=np.uint64))

            if analyzer is not None:
                analyzer.record(bytes_received=length)
            log.debug("msg_type: {}, length {}".format(msg_type, length))

            while len(data) < length:
                piece = socket.recv(min(2**14, length - len(data)))
                if len(piece) == 0:
                    return (MessageType._BYE, bytearray())
                data.extend(piece)

        return (msg_type, data)


class JsonBasedMessage:
    def set_data(self, data: dict = None):
        if data is not None:
            self.__dict__ = data
        return self

    def check(self):
        """Check the sanity of the message"""
        raise NotImplementedError("Pleas use a concrete subclass")

    def encode(self):
        self.check()
        return bytes(json.dumps(self.__dict__), encoding='utf8')

    def decode(self, data: bytearray):
        self.__dict__ = json.loads(data.decode('utf8'))
        self.check()
        return self


class ClientInfo(JsonBasedMessage):
    def check(self):
        assert (isinstance(self.data_len, int))
        assert (isinstance(self.histogram, list))


class TrainingConfig(JsonBasedMessage):
    def check(self):
        assert (isinstance(self.id, int))
        assert (isinstance(self.average_every, int))
        assert (isinstance(self.epochs, int))
        assert (isinstance(self.num_clients, int))
        assert (isinstance(self.batch_size, int))
        assert (isinstance(self.warmup_epochs, int))
        assert (isinstance(self.weight_decay, float))
        assert (isinstance(self.rotation_degree, float))
        assert (isinstance((self.momentum), float))
        assert (isinstance(self.patience, int))
        assert (isinstance(self.seed, int))
        assert (isinstance(self.server_ratio, float))
        assert (isinstance(self.client_ratio, float))
        assert (isinstance(self.server_rank, int))
        assert (isinstance(self.client_rank, int))
        assert (isinstance(self.global_momentum, float))

        # TODO: only certain compressors can perform momentum masking
        assert (isinstance(self.momentum_masking, bool))
        assert (isinstance(self.server_momentum_masking, bool))


class SyncInfos(JsonBasedMessage):
    def check(self):
        assert(isinstance(self.lr, float))
        assert(isinstance(self.seed, int))
        assert(isinstance(self.selected, bool))


class SecureForward(JsonBasedMessage):
    def check(self):
        assert(isinstance(self.src, int))
        assert(isinstance(self.dst, int))
        assert(isinstance(self.key, bytes))
        assert(isinstance(self.data, (bytes, bytearray)))

    def encode(self) -> bytearray:
        self.check()
        data = bytearray()
        data.extend(pack_int(self.src))
        data.extend(pack_int(self.dst))
        data.extend(pack_raw_data(self.key))
        data.extend(pack_raw_data(self.data))
        return data

    def decode(self, data):
        offset = Offset()
        self.src = unpack_int(data, offset)
        self.dst = unpack_int(data, offset)
        self.key = bytes(unpack_raw_data(data, offset))
        self.data = unpack_raw_data(data, offset)
        self.check()
        return self
