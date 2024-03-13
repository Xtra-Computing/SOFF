"""Communication protocols"""
import time
import json
import socket
import logging
import threading
from enum import Enum, unique
from typing import Dict, Optional
from munch import Munch
import numpy as np
from ..utils.logging import DataLogger
from ..compressors.compress_utils import (
    Offset, pack_int, pack_raw_data, unpack_int, unpack_raw_data)

# This file defines a custom socket comm protocol
log = logging.getLogger(__name__)


@unique
class MessageType(Enum):
    """Leading byte to identify the message type"""
    HANDSHAKE = b'h'
    METADATA = b'M'

    TRAINING_CONFIG = b't'
    MODEL = b'm'
    GRADIENT = b'g'
    INDICES = b'i'                # indices, for compression-related algs
    SYNC_INFOS = b's'             # for fedavg-like algorithm, metadatas
    CLIENT_INFOS = b'c'

    WEIGHTED_SUM_GRADIENT = b'w'  # for fednova, \(p_id_i^{(t-1)}\)
    STEP_WEIGHTS = b'S'           # for fednova, \(a_i in the paper\)

    RSA_PUBKEY = b'r'
    SECURE_FWD = b'f'             # forward message to another client

    LAYER_WEIGHTS = b'l'          # for fedma, weights of layers
    LAYER_GRADIENT = b'L'         # for fedma gradient of layers

    BYE = b'B'                    # shutdown signal, usually from srv to cli
    _BYE = b'b'                   # for dispatcher's internal use


# TODO: rename to DataTransceiver and use buffered async send/recv
class Protocol:
    meta_lock = threading.Lock()
    send_locks: Dict[socket.socket, threading.Lock] = {}
    recv_locks: Dict[socket.socket, threading.Lock] = {}
    """
    Ensure that only 1 thread sends/recvs from one socket,
    otherwise data will be scrambled
    """

    @staticmethod
    def send_data(
            sock: socket.socket, msg_type: MessageType, data: bytes,
            datalogger: Optional[DataLogger] = None,
            epochs: Optional[int] = None):
        """Send data to socket until all data sent"""
        assert (len(msg_type.value) == 1)

        header = bytearray()

        header.extend(msg_type.value)
        header.extend(np.uint64([len(data)]).tobytes())

        with Protocol.meta_lock:
            if sock not in Protocol.send_locks:
                Protocol.send_locks[sock] = threading.Lock()

        with Protocol.send_locks[sock]:
            log.debug("Sending data ->%s: %s (len=%s)",
                      sock.fileno(), msg_type, len(data))
            send_start = time.time()
            sock.sendall(header)
            sock.sendall(data)
            send_end = time.time()

        if datalogger is not None:
            datalogger.add_scalar("DataSent", len(data), epochs)
            datalogger.add_scalar(
                "DataSentTime", send_end - send_start, epochs)
        log.debug("Data sent")

    @staticmethod
    def recv_data(
            sock: socket.socket,
            datalogger: Optional[DataLogger] = None,
            epochs: Optional[int] = None):
        """Receive data from socket until a complete message is read"""
        log.debug("Receiving data")
        data = bytearray()
        piece = bytearray()
        with Protocol.meta_lock:
            if sock not in Protocol.recv_locks:
                Protocol.recv_locks[sock] = threading.Lock()

        with Protocol.recv_locks[sock]:
            recv_start = time.time()
            while len(piece) < 9:
                try:
                    subpiece = sock.recv(9 - len(piece))
                except ConnectionResetError:
                    return (MessageType._BYE, bytearray())

                if len(subpiece) == 0:
                    return (MessageType._BYE, bytearray())
                piece.extend(subpiece)

            msg_type = MessageType(bytes(piece[0:1]))
            length = int(np.frombuffer(piece[1:9], dtype=np.uint64))
            recv_end = time.time()

            log.debug("msg_type: %s, length %s", msg_type, length)
            if datalogger is not None:
                datalogger.add_scalar("DataReceived", length, epochs)
                datalogger.add_scalar(
                    "DataRecvTime", recv_end - recv_start, epochs)

            while len(data) < length:
                piece = bytearray(sock.recv(min(2**14, length - len(data))))
                if len(piece) == 0:
                    return (MessageType._BYE, bytearray())
                data.extend(piece)

        return (msg_type, data)


class Message:
    """The message object to send/receive"""

    def __init__(self, data: Munch = Munch()):
        self.data: Munch = data

    def set_data(self, dic: Optional[Dict] = None):
        """Set data for the message"""
        if dic is not None:
            self.data = Munch.fromDict(dic)
        return self

    def check(self):
        """Check the sanity of the message"""
        raise NotImplementedError

    def encode(self):
        """Serialize the message"""
        self.check()
        return bytes(json.dumps(self.data), encoding='utf-8')

    def decode(self, data: bytearray):
        """Deserialize the message"""
        self.data = Munch.fromDict(json.loads(data.decode('utf-8')))
        self.check()
        return self


class MetaData(Message):
    def check(self):
        pass


class ClientInfo(Message):
    def check(self):
        assert isinstance(self.data.data_len, int)


class FedMAClientInfo(ClientInfo):
    def check(self):
        super().check()
        assert isinstance(self.data.network_shape, dict)
        for key, val in self.data.network_shape.items():
            assert isinstance(key, str)
            assert isinstance(val, list)


class ForwardedConfig(Message):
    def check(self):
        assert isinstance(self.data.client_id, int)


class SyncInfos(Message):
    def check(self):
        assert isinstance(self.data.lr, float)
        assert isinstance(self.data.seed, int)
        assert isinstance(self.data.selected, bool)


class FedMASyncInfos(SyncInfos):
    def check(self):
        super().check()
        assert isinstance(self.data.frozen_layers, int)


class SecureForward(Message):
    def check(self):
        assert isinstance(self.data.src, int)
        assert isinstance(self.data.dst, int)
        assert isinstance(self.data.key, bytes)
        assert isinstance(self.data.data, (bytes, bytearray))

    def encode(self) -> bytearray:
        self.check()
        data = bytearray()
        data.extend(pack_int(self.data.src))
        data.extend(pack_int(self.data.dst))
        data.extend(pack_raw_data(self.data.key))
        data.extend(pack_raw_data(self.data.data))
        return data

    def decode(self, data):
        offset = Offset()
        self.data['src'] = unpack_int(data, offset)
        self.data['dst'] = unpack_int(data, offset)
        self.data['key'] = bytes(unpack_raw_data(data, offset))
        self.data['data'] = unpack_raw_data(data, offset)
        self.check()
        return self
