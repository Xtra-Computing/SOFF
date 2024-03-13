"""Base class of all FL servers schedulers"""
import os
import signal
import logging
from socket import SocketType
import threading
from abc import ABC, abstractclassmethod, abstractmethod
from typing import Callable, Dict, List, Optional, Type
from munch import Munch
from ...utils.logging import DataLogger
from ...utils.arg_parser import BaseConfParser, require
from ...communications.protocol import Protocol, MessageType, ForwardedConfig
from ...communications.dispatcher import ServerDispatcher


class BaseServerScheduler(ABC):
    """
    Scheduler for the server. Records client info and handles basic events.
    This scheduler requires the number of clients to be known in advance, so
    that it can properly fire start events and handle shutdown automatically.

    Before calling `start`, `register`
    """

    def __init__(
            self, cfg: Munch, client_cfg: Optional[Munch] = None,
            datalogger: Optional[DataLogger] = None) -> None:
        """
        Args:
            cfg: config passed to this module and its submodules.
            client_cfg: config to send to clients. If `None`, default to `cfg`.
            datalogger:
        """
        self.log = logging.getLogger(self.__class__.__name__)

        self.dispatcher = ServerDispatcher(cfg, datalogger=datalogger)

        self.cfg_for_client = client_cfg or cfg
        """Configs to send to the client upon receiving a handshake"""

        self.clients_socket_id_map: Dict[SocketType, int] = {}
        """Map socket to client_id"""
        self.clients_id_socket_map: Dict[int, SocketType] = {}
        """Map client_id to socket"""
        self.num_clients_connected = 0
        """Current number of clients connected"""
        self.clients_info_lock = threading.Lock()
        """Lock for updating client info"""

        self.cleanup_hooks: List[Callable[[], None]] = []
        """A list of cleanup hooks to execute at cleanup"""

        self.dispatcher.register_msg_event(
            MessageType.HANDSHAKE, self.process_handshake)
        self.dispatcher.register_msg_event(
            MessageType._BYE, self.process_bye)
        self.dispatcher.register_shutdown_event(
            self.process_dispatcher_shutdown)

        self.init_signal_handler()

    def start(self):
        self.dispatcher.start()

    @abstractclassmethod
    def conf_parser(cls) -> Type:
        raise NotImplementedError('conf_parser is not specified')

    @abstractmethod
    def process_handshake(self, socket, data):
        """Verify handshake message and assign client id"""
        raise NotImplementedError("process_handshake is not implemented.")

    @abstractmethod
    def process_bye(self, socket: SocketType, _):
        """
        Process `BYE` message from client.
        Determins how and when should the sever stop in a normal path.
        """
        raise NotImplementedError("process_bye is not implemented.")

    def process_dispatcher_shutdown(self):
        """Handles dispatcher abnormal shutdown"""
        self.log.warning("Exception in server. Stopping ...")
        self.cleanup()
        os._exit(1)

    def cleanup(self):
        """common cleanup hooks"""
        # Ignore exception at cleanup stage
        try:
            for hook in self.cleanup_hooks:
                hook()
        except Exception as exc:  # pylint: disable=broad-except
            self.log.exception(
                "Exception exceuting cleanup hook", exc_info=exc)
        finally:
            # cleanup and close all sockets gracefully,
            # in both exception-path and normal-path
            self.dispatcher.stop()

    def register_cleanup_hook(self, hook: Callable[[], None]) -> None:
        """
        Register a hook to be called when cleanup. Note: hook might be called
        from any thread (main/dispatcher/task scheduler).
        """
        self.cleanup_hooks.append(hook)

    def init_signal_handler(self):
        """Initializes handler for the SIGINT (Ctrl-C) signal"""
        signal.signal(signal.SIGINT, self.sigint_handler)

    def sigint_handler(self, *_, **__):
        """Handler for the SIGINT singal"""
        self.log.warning('SIGINT detected. Stopping everything...')
        self.cleanup()
        os._exit(0)

    def _schedule_unregister_client(self, socket: SocketType):
        if socket not in self.clients_socket_id_map:
            return
        client_id = self.clients_socket_id_map[socket]

        def __unregister_clients(socket, client_id):
            self.log.info(
                "Connection to client %s closed (fd=%s)",
                client_id, socket.fileno())
            with self.clients_info_lock:
                self.clients_socket_id_map.pop(socket)
                self.clients_id_socket_map.pop(client_id)
                self.num_clients_connected -= 1

        # Insert a task to the dispatcher to remove the clients
        self.dispatcher.insert_task_barrier()
        self.dispatcher.schedule_task(__unregister_clients, socket, client_id)
        self.dispatcher.insert_task_barrier()


class StaticBaseServerSchedulerConfParser(BaseConfParser):
    def __init__(self, *args, stag='', ltag='', **kwargs):
        super().__init__(*args, **kwargs)
        self.ss_args = self.add_argument_group("Server-Scheduler Configs")
        self.ss_args.add_argument(
            f'-ss{stag}.n', f'--server-scheduler{ltag}.num-endpoints',
            type=int, required=True, metavar='N',
            help="Number of endpoints (transceivers/clients) that connects to "
            "the server scheduler. Required for algs using static scheduler.")
        self.ss_args.add_argument(
            f'-ss{stag}.bi', f'--server-scheduler{ltag}.base-id',
            type=int, default=0, metavar='ID',
            help="Base client id. Endpoings connecting to this server scheduler "
            "will be assigned id starting from this number.")


@require('server.scheduler-num-endpoints')
class StaticBaseServerScheduler(BaseServerScheduler):
    """Requires num_clients, assigns endpoint ids statically & sequentially."""

    def __init__(self, cfg: Munch, datalogger=None) -> None:
        super().__init__(cfg, datalogger)

        self.base_id = cfg.server_scheduler.base_id
        self.num_clients = cfg.server_scheduler.num_endpoints
        """Total number of clients to connect"""

        self.event_all_clients_connected = threading.Event()
        """A One-shot event signaling the specified num of clients connected"""

    @classmethod
    def conf_parser(cls):
        return StaticBaseServerSchedulerConfParser

    def process_handshake(self, socket, data):
        if data != b'Hello!':
            self.log.warning("Handshake message verification failed!")
            return

        with self.clients_info_lock:
            # Generate config to send to client, assign client id
            msg = ForwardedConfig(self.cfg_for_client)

            client_id = next((
                i for i in range(self.base_id, self.base_id + self.num_clients)
                if i not in self.clients_id_socket_map), None)

            if client_id is None:
                self.log.warning(
                    "No more spare endpoints slots! (fd=%s)", socket.fileno())
                return

            msg.data['client_id'] = client_id
            self.clients_socket_id_map[socket] = msg.data.client_id
            self.clients_id_socket_map[msg.data.client_id] = socket

            self.log.info(
                "Client %s --> fd=%s ✔", msg.data.client_id, socket.fileno())

            self.dispatcher.schedule_task(
                Protocol.send_data, socket,
                MessageType.TRAINING_CONFIG, msg.encode())

            # if number of clients is enough, start training
            self.num_clients_connected += 1
            if self.num_clients_connected == self.num_clients:
                self.event_all_clients_connected.set()

    def process_bye(self, socket, _):
        # skip non-registered clients
        if socket not in self.clients_socket_id_map:
            return

        with self.clients_info_lock:
            # if all clients leaves while training already started, it means
            # an abnormal state. force shutdown. (normal shutdown should be
            # handled by subclass protocol and initiated by the server.)
            if self.event_all_clients_connected.is_set() and \
                    self.num_clients_connected <= 1:
                self.log.info("All clients are gone, shutting down...")
                self.cleanup()
                # this function is called from dispatcher thread, so force
                # stopping the main thread is necessary. see:
                # https://stackoverflow.com/questions/1489669
                os._exit(1)

        self._schedule_unregister_client(socket)


class DynamicBaseServerScheduler(BaseServerScheduler):
    """
    Doesn't require a fixed num_clients.
    Requires manual shutdown by self.cleanup() and os._exit(1)
    """

    def __init__(self, cfg: Munch, datalogger=None) -> None:
        super().__init__(cfg, datalogger)
        self.__cur_max_client_id = 0

    @classmethod
    def conf_parser(cls):
        return BaseConfParser

    def process_handshake(self, socket, data):
        if data != b'Hello!':
            self.log.warning("Handshake message verification failed!")
            return

        with self.clients_info_lock:
            # Generate config to send to client, assign client id
            msg = ForwardedConfig(self.cfg_for_client)

            msg.data['client_id'] = self.__cur_max_client_id
            self.clients_socket_id_map[socket] = msg.data.client_id
            self.clients_id_socket_map[msg.data.client_id] = socket
            self.log.info(
                "Client %s --> fd=%s ✔", msg.data.client_id, socket.fileno())

            self.dispatcher.schedule_task(
                Protocol.send_data, socket,
                MessageType.TRAINING_CONFIG, msg.encode())

            self.__cur_max_client_id += 1
            self.num_clients_connected += 1

    def process_bye(self, socket, _):
        self._schedule_unregister_client(socket)
