"""Low-level client-server message dispatcher via epoll"""
import os
import time
import queue
import socket
import select
import pathlib
import logging
import threading
from typing import Any, Callable, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, wait
from munch import Munch
from eventfd import EventFD
from ..communications.protocol import MessageType, Protocol
from ..utils.arg_parser import BaseConfParser
from ..utils.logging import DataLogger

log = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, cfg):
        self.bandwidth = cfg.dispatcher.rate_limit
        self.latency = cfg.dispatcher.rate_limiter.latency
        self.buffer_size = cfg.dispatcher.rate_limiter.buffer_size

        # Burst control. send/recv within this time interval is rate limited
        self.burst_interval = 1.5 * self.buffer_size / self.bandwidth

        self._sent_lock = threading.Lock()
        self._bytes_sent = 0
        self._send_started = time.time()

        self._received_lock = threading.Lock()
        self._bytes_received = 0
        self._recv_started = time.time()

    def send_wait(self):
        """Calculate time to wait before next send"""
        duration = time.time() - self._send_started
        if duration > self.burst_interval:
            with self._sent_lock:
                self._send_started = time.time()
                self._bytes_sent = 0
        else:
            required_duration = self._bytes_sent / self.bandwidth
            time.sleep(max(required_duration - duration, self.latency))

    def recv_wait(self):
        """Calculate time to wait before next recv"""
        duration = time.time() - self._recv_started
        if duration > self.burst_interval:
            with self._received_lock:
                self._recv_started = time.time()
                self._bytes_received = 0
        else:
            required_duration = self._bytes_received / self.bandwidth
            time.sleep(max(required_duration - duration, self.latency))

    def sent_bytes(self, num_bytes):
        """Track bytes sent"""
        with self._sent_lock:
            self._bytes_sent += num_bytes

    def received_bytes(self, num_bytes):
        """Track bytes received"""
        with self._received_lock:
            self._bytes_received += num_bytes


class LimitedSocket(socket.socket):
    """Wrapper of socket with a rate limiter"""

    def __init__(self, limiter: RateLimiter, *args, **kwargs):
        # rate limiting variables
        self.rate_limiter = limiter
        self.buffer_size = limiter.buffer_size
        super().__init__(*args, **kwargs)

    def sendall(self, data: bytearray):
        data_sent = 0
        while data_sent < len(data):
            self.rate_limiter.send_wait()
            if data_sent + self.buffer_size >= len(data):
                super().sendall(data[data_sent:])
                data_sent += len(data)
                self.rate_limiter.sent_bytes(len(data))
            else:
                super().sendall(data[data_sent:data_sent + self.buffer_size])
                data_sent += self.buffer_size
                self.rate_limiter.sent_bytes(self.buffer_size)

    def recv(self, bufsize, *args, **kwargs):
        self.rate_limiter.recv_wait()
        piece = super().recv(min(self.buffer_size, bufsize), *args, **kwargs)
        self.rate_limiter.received_bytes(len(piece))
        return piece

    def accept(self):
        """ Overriden method, returns controlled socket instead of socket """
        fileno, addr = self._accept()
        sock = LimitedSocket(
            self.rate_limiter,
            self.family, self.type, self.proto, fileno=fileno)
        if socket.getdefaulttimeout() is None and self.gettimeout():
            sock.setblocking(True)
        return sock, addr


class ServerDispatcher:
    """A __nonblocking__ server that executes in another thread"""

    # TODO: This server can be further decoupled to a pure socket server and a
    # task scheduler

    def __init__(self, cfg: Munch, datalogger=None):
        self.socket_type = cfg.dispatcher.socket_type
        self.addr = cfg.dispatcher.address
        self.server = None

        rate_limit = cfg.dispatcher.rate_limit
        self.rate_limiter: Optional[RateLimiter] = None \
            if rate_limit <= 0 else RateLimiter(cfg)
        """The server socket rate-limiter"""

        self.comm_thread = threading.Thread(target=self.__comm_scheduler)
        """The master thread for communication"""

        self.schedule_thread = threading.Thread(target=self.__task_scheduler)
        """The master thread for scheduling all tasks"""

        self.thread_pool = ThreadPoolExecutor(cfg.dispatcher.num_threads)
        """Thread pool to handle all jobs (to avoid blocking callback)"""
        self.task_queue: queue.Queue = queue.Queue()
        """task to queue the jobs (to guarantee execution sequence)"""

        self.registered_fds: Dict[int, EventFD] = {}
        """a dict of events to handle, key is fileno, val is file descriptor"""
        self.registered_fd_callbacks: Dict[int, Callable] = {}
        """a dict of events to handle, key is fileno, val is callback"""
        self.registered_msg_callbacks: Dict[MessageType, Callable] = {}
        """a dict of events to handle, key is msg type, val is callback"""
        self.registered_shutdown_callbacks: List[Callable] = []
        """a list of callbacks to handle"""

        self.epoll = select.epoll()
        """use epoll to deal with events"""
        self.event_shutdown = EventFD()
        """signal that everything is done"""

        self.clients: Dict[int, socket.socket] = {}
        """dict to note all connected clients. key is fileno, val is socket"""

        self.datalogger = datalogger
        """logger to record sending/receiving statistics"""

    def register_fd_event(
            self, eventfd: EventFD, callback: Callable[[EventFD], None]):
        """
        eventfd: A file descriptor to trigger the custom event.
        callback: A callback function that will be immediately executed.
            The associated eventfd that triggers this event will be passed
            as the only argument.

        This function should be *non-blocking*.  use `schedule_task`
        in this function to schedule long-running tasks.
        """
        fileno = eventfd.fileno()
        assert fileno is not None
        self.epoll.register(fileno, select.EPOLLIN | select.EPOLLET)
        self.registered_fds[fileno] = eventfd
        self.registered_fd_callbacks[fileno] = callback

    def register_msg_event(
            self, msg_type: MessageType,
            callback: Callable[[socket.socket, bytearray], None]):
        """
        msg_type: One of the MessageType enumerate
        callback: A callback function that will be immediately executed.
            The callbacks takes two arguments
            1. the associated socket that receives this message
            2. the received message (without header).

        This function should be *non-blocking*.  use `schedule_task`
        in this function to schedule long-running tasks.
        """
        assert isinstance(msg_type, MessageType)
        self.registered_msg_callbacks[msg_type] = callback

    def register_shutdown_event(self, callback: Callable[[], Any]):
        """
        callback: A callback funtion that will be immediatly executed when
            server is shutting done. No arguments will be passed.

        This function should be *non-blocking*.
        """
        self.registered_shutdown_callbacks.append(callback)

    def start(self):
        """Listen on the socket and start handling messages"""
        # Initialize socket ###################################################
        if self.socket_type == 'unix':
            # Test existing socket file
            try:
                os.unlink(self.addr)
            except OSError:
                if os.path.exists(self.addr):
                    log.exception("Unix socket file already exist.")
                    raise
            # Create directory for socket
            socket_dir = pathlib.Path(self.addr).parent
            if not socket_dir.exists():
                socket_dir.mkdir(parents=True, exist_ok=True)
            # Listen on socket
            self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) \
                if self.rate_limiter is None \
                else LimitedSocket(
                self.rate_limiter, socket.AF_UNIX, socket.SOCK_STREAM)
            self.server.bind(self.addr)
        elif self.socket_type == 'tcp':
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) \
                if self.rate_limiter is None \
                else LimitedSocket(
                    self.rate_limiter, socket.AF_INET, socket.SOCK_STREAM)
            host, port = tuple(self.addr.split(':'))
            port = int(port)
            self.server.bind((host, port))
        else:
            raise Exception("Unknown socket type")

        self.server.setblocking(False)
        self.server.listen()

        # start communication thread and task scheduler thread
        self.comm_thread.start()
        self.schedule_thread.start()

    def stop(self):
        """Send a signal to stop the server gracefully"""
        self.event_shutdown.set()
        # avoid being waiting the self on some thread.
        if threading.get_ident() != self.comm_thread.ident:
            self.comm_thread.join(timeout=5)
        if threading.get_ident() != self.schedule_thread.ident:
            self.schedule_thread.join(timeout=5)

    def schedule_task(self, func, *args, **kwargs):
        """Schedule a random task"""
        self.task_queue.put([func, args, kwargs])

    def schedule_broadcast(
            self, msg_type: MessageType, data: bytearray,
            datalogger: Optional[DataLogger] = None,
            epochs: Optional[int] = None):
        """Schedule tasks to broadcast message to all connected clients"""
        for sock in self.clients.values():
            self.task_queue.put([
                Protocol.send_data,
                [sock, msg_type, data, datalogger, epochs], {}
            ])

    def insert_task_barrier(self):
        """
        insert a barrier to the task queue. When barrier is encountered,
        the task scheduler will wait for all previous scheduled task to end
        before executing new tasks, which is used for guarantee the execution
        sequence. The execution order between two barriers are not guaranteed.
        """
        self.task_queue.put(None)

    def __comm_scheduler(self):
        """
        File descriptor events and message scheduler, meant to be run in
        another thread.
        """
        assert self.server is not None

        def process_event(fileno, event) -> bool:
            """Return: True - continue. False - stop"""
            if fileno == self.server.fileno():
                # accept and register client connection
                conn, _ = self.server.accept()
                # we want client sockets to be blocking
                conn.setblocking(True)
                self.clients[conn.fileno()] = conn
                self.epoll.register(conn, select.EPOLLIN)
                log.debug("Connection to fd=%s established", conn.fileno())
            elif fileno in self.registered_fds:
                self.registered_fd_callbacks[fileno](
                    self.registered_fds[fileno])
                self.registered_fds[fileno].clear()
            elif fileno == self.event_shutdown.fileno():
                # The normal-shutdown path
                for callback in self.registered_shutdown_callbacks:
                    callback()
                log.debug("Shutting down ...")
                return False
            elif event & select.EPOLLIN:
                log.debug("Handling data (fd=%s)", fileno)
                # avoid receiving data from disconnected clients
                # (which raises exception)
                if fileno not in self.clients:
                    return True
                msg_type, data = Protocol.recv_data(
                    self.clients[fileno], self.datalogger)
                if msg_type in self.registered_msg_callbacks:
                    self.registered_msg_callbacks[msg_type](
                        self.clients[fileno], data)
                else:
                    log.error("Unknown message type")
                    return False
                # additionally, handle a special BYE message
                # (which could be generated by receiving empty message)
                if msg_type == MessageType._BYE:
                    self.epoll.unregister(self.clients[fileno])
                    self.clients[fileno].close()
                    self.clients.pop(fileno)
            else:
                log.error("Unknown epoll event")
            return True

        self.epoll.register(self.server.fileno(), select.EPOLLIN)
        self.epoll.register(self.event_shutdown.fileno(),
                            select.EPOLLIN | select.EPOLLET)
        try:
            while True:
                events = self.epoll.poll()
                for fileno, event in events:
                    if not process_event(fileno, event):
                        return
        except Exception as exc:
            # The exception-shutdown path
            log.error("Exception caught in dispatcher (comm thread).")
            log.exception(exc)

            # also signal the task_scheduler thread to exit
            self.event_shutdown.set()

            # execute shutdown hook (inform main thread to exit through hook)
            for callback in self.registered_shutdown_callbacks:
                callback()
        finally:
            log.debug("Cleaning up all connections")
            self.epoll.unregister(self.server.fileno())
            self.epoll.close()
            for sock in self.clients.values():
                sock.close()
            self.server.close()
            if self.socket_type == 'unix':
                os.unlink(self.addr)

    def __task_scheduler(self):
        """task scheduler, meant to be run in another thread. """

        running_tasks = set()
        # main event loop to schedule tasks and handle shutdown event
        while True:
            # do garbage collection before scheduling a task
            finished_tasks = set(
                filter(lambda task: task.done(), running_tasks))
            for task in finished_tasks:
                # check exceptions in tasks immediatly
                exc_info = task.exception(timeout=0)
                if exc_info is not None:
                    # The exception-shutdown path
                    log.exception("Exception in dispatcher (task thread).",
                                  exc_info=exc_info)
                    self.event_shutdown.set()
                    return
            running_tasks = set(
                filter(lambda task: not task.done(), running_tasks))

            # try to get a task, while regularly probe for a exit signal
            callback = None
            if self.event_shutdown.is_set():
                # The normal-shutdown path
                return

            try:
                # the timeout here controls the probing frequency
                callback = self.task_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                continue

            # schedule a task
            if callback is None:
                # wait for all running tasks to complete
                wait(running_tasks)
                for result in running_tasks:
                    if result.exception() is not None:
                        log.exception("Exception in task.")
                        raise result.exception()
            elif isinstance(callback, list):
                # perform scheduled task
                assert (len(callback) == 3)
                future = self.thread_pool.submit(callback[0], *callback[1],
                                                 **callback[2])
                running_tasks.add(future)
            else:
                log.error("Callback format not recognized")


class ClientDispatcher:
    """A __blocking__ socket client"""

    def __init__(self, cfg: Munch):
        """socket_type: 'unix' or 'tcp'"""
        self.socket_type = cfg.dispatcher.socket_type
        self.addr = cfg.dispatcher.address
        self.client: socket.socket
        """The client socket"""

    def start(self, retry=300, retry_interval=1):
        """Start client dispatcher and try connect to the sever dispatcher"""
        # Retry connecting to the server until connected, or max retry exceeded
        while retry > 0:
            retry = retry - 1
            if self.socket_type == 'unix':
                self.client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    self.client.connect(self.addr)
                except Exception as e:
                    log.warning(e)
                    log.info("retrying ...")
                    time.sleep(retry_interval)
                    continue
                break
            elif self.socket_type == 'tcp':
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                host, port = self.addr.split(':')
                port = int(port)
                try:
                    self.client.connect((host, port))
                except ConnectionRefusedError as e:
                    log.warning(e)
                    log.info("retrying ...")
                    time.sleep(retry_interval)
                    continue
                break
            else:
                raise Exception("Unknown socket type")

        if retry == 0:
            raise Exception("Failed to connect to server")

        self.client.setblocking(True)

    def recv_msg(self):
        """
        Receive message in a blocking manner, and call callback function after
        Message is received
        """
        return Protocol.recv_data(self.client)

    def send_msg(
        self, msg_type: MessageType, data: bytes,
            datalogger: Optional[DataLogger] = None,
            epochs: Optional[int] = None):
        """
        Send message to server in a blocking manner
        """
        return Protocol.send_data(
            self.client, msg_type, data, datalogger, epochs)


class DispatcherConfParserAdapter(BaseConfParser):
    """Parse common configs for both server and clients"""

    def add_dispatcher_args(self, stag='', ltag=''):
        dispatcher_args = self.add_argument_group("Dispather Configs (S,C)")

        dispatcher_args.add_argument(
            f'-dc{stag}.s', f'--dispatcher{ltag}.socket-type',
            default='unix', help='<unix|tcp>', metavar='TYPE')

        dispatcher_args.add_argument(
            f'-dc{stag}.a', f'--dispatcher{ltag}.address',
            default='/tmp/soff/fed-comm.sock', type=str, metavar='ADDR',
            help="Socket address (unix socket only for local simulation,"
            "for cross-machine support, use tcp socket)")


class ServerDispatcherConfParserAdapater(BaseConfParser):
    """Pasre configs for server communication dispathers"""

    def add_server_dispatcher_args(self, stag='', ltag=''):
        dispatcher_args = self.add_argument_group(
            "Server Dispather Configs (S)")
        dispatcher_args.add_argument(
            f'-dc{stag}.n', f'--dispatcher{ltag}.num-threads',
            default=5, type=int, metavar='N',
            help="Number of thread for sending model and receiving gradient."
            " Should not exceed 2 * num-cache, otherwise waste resources.")
        dispatcher_args.add_argument(
            f'-dc{stag}.rl', f'--dispatcher{ltag}.rate-limit',
            default=0, type=int, metavar='LIMIT',
            help="send/recv rate limit, in Bytes/sec. <=0 for no limit")
        dispatcher_args.add_argument(
            f'-dc{stag}.rl.l', f'--dispatcher{ltag}.rate-limiter.latency',
            default=0, type=float, metavar='LATENCY',
            help="latency for the rate limiter, in sec. <=0 for no latency."
            "requires dispatcher.rate-limit > 0")
        dispatcher_args.add_argument(
            f'-dc{stag}.rl.bs', f'--dispatcher{ltag}.rate-limiter.buffer-size',
            default=1024*200, type=int, metavar='LATENCY',
            help="buffer size for the rate limiter's burst control, in bytes."
            "requires dispatcher.rate-limit > 0")
        self.register_cfg_dep(
            f"dispatcher.{ltag.replace('/', '.')}.rate-limiter",
            lambda c: c.dispatcher.rate_limit > 0)


class ServerDispatcherConfParser(
        ServerDispatcherConfParserAdapater, DispatcherConfParserAdapter):
    def __init__(self, *args, stag='', ltag='', **kwargs):
        super().__init__(*args, **kwargs)
        self.add_dispatcher_args(stag, ltag)
        self.add_server_dispatcher_args(stag, ltag)


class ClientDispatcherConfParserAdapter(BaseConfParser):
    """Pasre configs for client communication dispathers"""

    def add_client_dispatcher_args(self, stag='', ltag=''):
        dispatcher_args = self.add_argument_group(
            "Client Dispather Configs (C)")
        dispatcher_args.add_argument(
            f'-dc{stag}.r', f'--dispatcher{ltag}.retry',
            default=300, type=int, metavar='RETRY',
            help="Number of retries to connect to the server before abort")
        dispatcher_args.add_argument(
            f'-dc{stag}.ri', f'--dispatcher{ltag}.retry-interval',
            default=1, type=int, metavar='INTERVAL',
            help="Intervals between two connection retries")


class ClientDispatcherConfParser(
        ClientDispatcherConfParserAdapter, DispatcherConfParserAdapter):

    def __init__(self, *args, stag='', ltag='', **kwargs):
        super().__init__(*args, **kwargs)
        self.add_dispatcher_args(stag, ltag)
        self.add_client_dispatcher_args(stag, ltag)
