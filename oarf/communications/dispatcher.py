# Low-level client-server message scheduler via epoll
import os
import time
import queue
import socket
import select
import logging
import threading
from eventfd import EventFD
from oarf.metrics.analyzer import Analyzer
from concurrent.futures import ThreadPoolExecutor, wait
from oarf.communications.protocol import MessageType, Protocol

log = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, bandwidth=12_500_000, latency=0,
                 buffer_size=1024*200):
        self.bandwidth = bandwidth
        self.latency = latency
        self.buffer_size = buffer_size

        # Burst control. send/recv within this time interval is rate limited
        self.burst_interval = 1.5 * buffer_size / bandwidth

        self._sent_lock = threading.Lock()
        self._bytes_sent = 0
        self._send_started = time.time()

        self._received_lock = threading.Lock()
        self._bytes_received = 0
        self._recv_started = time.time()

    def send_wait(self):
        duration = time.time() - self._send_started
        if duration > self.burst_interval:
            with self._sent_lock:
                self._send_started = time.time()
                self._bytes_sent = 0
        else:
            required_duration = self._bytes_sent / self.bandwidth
            time.sleep(max(required_duration - duration, self.latency))

    def recv_wait(self):
        duration = time.time() - self._recv_started
        if duration > self.burst_interval:
            with self._received_lock:
                self._recv_started = time.time()
                self._bytes_received = 0
        else:
            required_duration = self._bytes_received / self.bandwidth
            time.sleep(max(required_duration - duration, self.latency))

    def sent_bytes(self, num_bytes):
        with self._sent_lock:
            self._bytes_sent += num_bytes

    def received_bytes(self, num_bytes):
        with self._received_lock:
            self._bytes_received += num_bytes


class LimitedSocket(socket.socket):
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
        fd, addr = self._accept()
        sock = LimitedSocket(
            self.rate_limiter,
            self.family, self.type, self.proto, fileno=fd)
        if socket.getdefaulttimeout() is None and self.gettimeout():
            sock.setblocking(True)
        return sock, addr


class ServerDispatcher:
    """An **nonblocking** server that executes in another thread"""

    # TODO: This server can be further decoupled to a pure socket server and a
    # task scheduler

    def __init__(self, socket_type, addr: str, num_clients, num_threads=5,
                 rate_limit=0, analyzer: Analyzer = None):
        """socket_type: 'unix' or 'tcp'"""
        self.socket_type = socket_type
        self.addr = addr
        self.server = None
        self.num_clients = num_clients

        if rate_limit == 0:
            self.rate_limiter = None
        else:
            assert rate_limit > 0
            self.rate_limiter = RateLimiter(rate_limit)
        """The server socket"""

        self.comm_thread = threading.Thread(target=self.__comm_scheduler)
        """The master thread for communication"""

        self.schedule_thread = threading.Thread(target=self.__task_scheduler)
        """The master thread for scheduling all tasks"""

        self.thread_pool = ThreadPoolExecutor(num_threads)
        """Thread pool to handle all jobs (to avoid blocking callback)"""
        self.task_queue = queue.Queue()
        """task to queue the jobs (to guarantee execution sequence)"""

        self.registered_fds = {}
        """a dict of events to handle, key is fileno, val is file descriptor"""
        self.registered_fd_callbacks = {}
        """a dict of events to handle, key is fileno, val is callback"""
        self.registered_msg_callbacks = {}
        """a dict of events to handle, key is msg type, val is callback"""
        self.registered_shutdown_callbacks = []
        """a list of callbacks to handle"""

        self.epoll = select.epoll()
        """use epoll to deal with events"""
        self.event_shutdown = EventFD()
        """signal that everything is done"""

        self.clients = dict()
        """dict to note all connected clients. key is fileno, val is socket"""

        self.analyzer = analyzer
        """analyzer to record receiving time"""

    def fileno_to_socket(self, fileno: int):
        return self.clients[fileno] if fileno in self.clients else None

    def register_fd_event(self, eventfd: EventFD, callback):
        """
        eventfd: A file descriptor to trigger the custom event.
        callback: A callback function that will be immediately executed.
                  The associated eventfd that triggers this event will be
                  passed as the only argument.

                  This function should be *non-blocking*.  use `schedule_task`
                  in this function to schedule long-running tasks.
        """
        self.epoll.register(eventfd.fileno(), select.EPOLLIN | select.EPOLLET)
        self.registered_fds[eventfd.fileno()] = eventfd
        self.registered_fd_callbacks[eventfd.fileno()] = callback

    def register_msg_event(self, msg_type: MessageType, callback):
        """
        msg_type: One of the MessageType enumerate
        callback: A callback function that will be immediately executed.
                  The associated socket (of the connected clients) which
                  triggers this event will be passed as the 1st argument, and
                  the received message (without header) will be passed as the
                  second argument.

                  This function should be *non-blocking*.  use `schedule_task`
                  in this function to schedule long-running tasks.
        """
        assert (isinstance(msg_type, MessageType))
        self.registered_msg_callbacks[msg_type] = callback

    def register_shutdown_event(self, callback):
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
            try:
                os.unlink(self.addr)
            except OSError:
                if os.path.exists(self.addr):
                    log.exception("Unix socket file already exist.")
            if self.rate_limiter is None:
                self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            else:
                self.server = LimitedSocket(
                    self.rate_limiter, socket.AF_UNIX, socket.SOCK_STREAM)
            self.server.bind(self.addr)
        elif self.socket_type == 'tcp':
            if self.rate_limiter is None:
                self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            else:
                self.server = LimitedSocket(
                    self.rate_limiter, socket.AF_INET, socket.SOCK_STREAM)
            host, port = tuple(self.addr.split(':'))
            port = int(port)
            self.server.bind((host, port))
        else:
            raise Exception("Unknown socket type")

        self.server.setblocking(False)
        self.server.listen(self.num_clients)

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

    def schedule_task(self, fn, *args, **kwargs):
        """Schedule a random task"""
        self.task_queue.put([fn, args, kwargs])

    def schedule_broadcast(self, msg_type: MessageType, data: bytearray,
                           analyzer: Analyzer = None):
        """Schedule tasks to broadcast message to all connected clients"""
        for fileno in self.clients:
            if analyzer is not None:
                self.task_queue.put([
                    Protocol.send_data, [], {
                        'socket': self.clients[fileno],
                        'msg_type': msg_type, 'data': data,
                        'analyzer': analyzer.send_data_parallel()
                    }
                ])
            else:
                self.task_queue.put([
                    Protocol.send_data, [], {
                        'socket': self.clients[fileno],
                        'msg_type': msg_type, 'data': data
                    }
                ])

    def insert_task_barrier(self):
        """
        insert a barrier to the task queue. When barrier is encountered,
        the task scheduler will wait for all previous scheduled task to end
        before executing new tasks, which is used for guarantee the execution
        sequence. The execution order between two barriers are not guaranteed.
        """
        self.task_queue.put(None)

    # TODO: add error handling

    def __comm_scheduler(self):
        """
        File descriptor events and message scheduler, meant to be run in
        another thread.
        """
        self.epoll.register(self.server.fileno(), select.EPOLLIN)
        self.epoll.register(self.event_shutdown.fileno(),
                            select.EPOLLIN | select.EPOLLET)
        try:
            while True:
                events = self.epoll.poll()
                for fileno, event in events:
                    if fileno == self.server.fileno():
                        # accept and register client connection
                        conn, addr = self.server.accept()
                        # we want client sockets to be blocking
                        conn.setblocking(True)
                        self.clients[conn.fileno()] = conn
                        self.epoll.register(conn, select.EPOLLIN)
                        log.debug("Connection to fd={} established".format(
                            conn.fileno()))
                    elif fileno in self.registered_fds:
                        self.registered_fd_callbacks[fileno](
                            self.registered_fds[fileno])
                        self.registered_fds[fileno].clear()
                    elif fileno == self.event_shutdown.fileno():
                        for callback in self.registered_shutdown_callbacks:
                            callback()
                        log.debug("Shutting down ...")
                        return
                    elif event & select.EPOLLIN:
                        log.debug("Handling data (fd={})".format(fileno))
                        # avoid receiving data from disconnected clients
                        # (which raises exception)
                        if fileno not in self.clients:
                            continue
                        # TODO: switch to multi-thread receive?
                        if self.analyzer is not None:
                            with self.analyzer.recv_data_parallel() as ra:
                                msg_type, data = Protocol.recv_data(
                                    self.clients[fileno], ra)
                        else:
                            msg_type, data = Protocol.recv_data(
                                self.clients[fileno])

                        if msg_type in self.registered_msg_callbacks:
                            self.registered_msg_callbacks[msg_type](
                                self.clients[fileno], data)
                        else:
                            log.error("Unknown message type")
                            return
                        # additionally, handle a special BYE message
                        # (which could be generated by receiving empty message)
                        if msg_type == MessageType._BYE:
                            self.epoll.unregister(self.clients[fileno])
                            self.clients[fileno].close()
                            self.clients.pop(fileno)
                    else:
                        log.error("Unknown epoll event")
        finally:
            log.debug("Cleaning up all connections")
            self.epoll.unregister(self.server.fileno())
            self.epoll.close()
            for fileno in self.clients:
                self.clients[fileno].close()
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
                    log.exception("Exception in task.", exc_info=exc_info)
                    self.event_shutdown.set()
                    return
            running_tasks = set(
                filter(lambda task: not task.done(), running_tasks))

            # try to get a task, while regularly probe for a exit signal
            callback = None
            if self.event_shutdown.is_set():
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
    """A **blocking** socket client"""

    def __init__(self, socket_type, addr: str):
        """socket_type: 'unix' or 'tcp'"""
        self.socket_type = socket_type
        self.addr = addr
        self.client = None
        """The client socket"""

    def start(self, retry=300, retry_interval=1):
        # TODO: replace this with semaphore or something...
        # probing is too inefficient
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
                except Exception as e:
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

    def send_msg(self, msg_type: MessageType, data: bytearray):
        """
        Send message to server in a blocking manner
        """
        return Protocol.send_data(self.client, msg_type, data)
