import os
import sys
import json
import signal
import threading
from functools import partial
from oarf.security import rsa
from oarf.metrics.analyzer import ServerAnalyzer
from torch.utils.data import DataLoader
from oarf.utils.logging import Logger
from oarf.utils.training import Deterministic
from oarf.utils.arg_parser import ServerArgParser
from oarf.datasets.datasets import create_dataset
from oarf.compressors.compressors import Compressor
from oarf.communications.dispatcher import ServerDispatcher
from oarf.communications.protocol import (
    Protocol, MessageType, TrainingConfig, SecureForward)


# Parse arguments and initialize application ##################################
class BaseSever(Logger, Deterministic, Compressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, logger_name=self.__class__.__name__)
        self.log.info('\n' + json.dumps(kwargs, indent=2))
        self.analyzer = ServerAnalyzer()

        # avoid passing garbage to clients
        self.args_for_client = ServerArgParser.get_args_for_clients(kwargs)
        self.init_dataset(**kwargs)
        self.init_dispatcher(**kwargs)
        self.init_secure_aggregation(**kwargs)

        self.cleanup = partial(self.__cleanup, kwargs)
        self.init_signal_handler(kwargs)

    @classmethod
    def start(cls):
        argparser = ServerArgParser()
        args = argparser.parse_args()
        server = cls(**args.__dict__)
        server.dispatcher.start()
        server.init_training(**args.__dict__)
        server.start_training(**args.__dict__)
        server.cleanup()

    def init_training(self, **_):
        raise NotImplementedError

    def start_training(self, **_):
        raise NotImplementedError

    def init_dispatcher(
            self, socket_type, address, num_clients, num_threads,
            batch_size, rate_limit, **_):

        self.dispatcher = ServerDispatcher(
            socket_type, address, num_clients, num_threads=num_threads,
            rate_limit=rate_limit, analyzer=self.analyzer)

        self.clients_socket_id_map = dict()
        self.clients_id_socket_map = dict()
        self.num_clients_connected = 0
        """the number of clients connected"""
        self.clients_info_lock = threading.Lock()

        self.event_all_clients_connected = threading.Event()
        """one-shot event signaling the specified num of clients connected"""

        self.sem_client_event = threading.Semaphore(0)
        """signals that an client event occurred.
        e.g. a client's data is received and parsed, or clients are gone.
        """

        self.dispatcher.register_msg_event(
            MessageType.HANDSHAKE,
            partial(self.process_handshake, num_clients=num_clients))
        self.dispatcher.register_msg_event(
            MessageType._BYE,
            partial(self.process_bye, num_clients=num_clients))
        self.dispatcher.register_shutdown_event(
            partial(self.process_dispatcher_shutdown, num_clients, batch_size))

    def init_dataset(self, datasets, test_datasets,
                     data_splitting, num_clients, batch_size, **_):
        # Initialize test dataset #############################################
        self.eval_dataset = create_dataset(
            datasets, data_splitting=data_splitting, mode='eval',
            num_clients=num_clients, client_id=None)
        # self.test_dataset = create_dataset(
        #     datasets, data_splitting=data_splitting, mode='test',
        #     num_clients=num_clients, client_id=None)
        test_datasets = ([','.join(datasets)] if test_datasets is None
                         else test_datasets)
        self.test_datasets = {
            tdss: create_dataset(
                datasets=tdss.split(','), data_splitting='iid',
                mode='test', num_clients=1, client_id=None)
            for tdss in test_datasets
        }

        self.eval_loader = DataLoader(
            self.eval_dataset, batch_size, shuffle=False)
        # self.test_loader = DataLoader(
        #     self.test_dataset, batch_size, shuffle=False)
        self.test_loaders = {
            k: DataLoader(v, batch_size, shuffle=False)
            for k, v in self.test_datasets.items()
        }

    def init_secure_aggregation(self, secure_aggregation_method,
                                secret_splitting_num, num_clients, *_, **__):

        if secure_aggregation_method is None:
            self.secure_aggregation = False
        elif secure_aggregation_method == 'SS':
            assert 2 <= secret_splitting_num <= num_clients
            self.secret_splitting_num = secret_splitting_num
            self.secure_aggregation = True
            self.rsa_keyring_lock = threading.Lock()
            self.rsa_keyring = rsa.KeyChain()
            self.dispatcher.register_msg_event(
                MessageType.RSA_PUBKEY,
                partial(self.register_client_pubkey, num_clients=num_clients))
            self.dispatcher.register_msg_event(
                MessageType.GRAD_SPLIT, self.forward_grad_split)
        else:
            raise NotImplementedError("Not implemented yet")

    def register_client_pubkey(self, socket, data, /, num_clients):
        with self.rsa_keyring_lock:
            self.rsa_keyring.add_key(
                self.clients_socket_id_map[socket],
                rsa.Key().deseralize_pubkey(data))

            # broadcast all pubkey if pubkey of all clients are received
            if len(self.rsa_keyring) == num_clients:
                self.dispatcher.schedule_broadcast(
                    MessageType.RSA_PUBKEY,
                    data=self.rsa_keyring.seralize_pubkeys(),
                    analyzer=self.analyzer)

    def forward_grad_split(self, socket, data):
        # simply forward msg to required destination
        msg = SecureForward().decode(data)
        self.log.info("Forwarding message {} -> {}".format(msg.src, msg.dst))
        self.dispatcher.schedule_task(
            Protocol.send_data, self.clients_id_socket_map[msg.dst],
            MessageType.GRAD_SPLIT, data)

    def init_signal_handler(self, args):
        # handle SIGINT (Ctrl-C)
        signal.signal(signal.SIGINT, self.sigint_handler)

    def process_handshake(self, socket, data, /, num_clients):
        # verify handshake message
        if data != b'Hello!':
            self.log.warning("Handshake message verification failed!")
            return

        with self.clients_info_lock:
            # Generate config to send to client, decide client id
            config = TrainingConfig().set_data(self.args_for_client)
            config.id = self.num_clients_connected
            self.clients_socket_id_map[socket] = config.id
            self.clients_id_socket_map[config.id] = socket
            self.log.info("Client {} --> fd={} ✔".format(
                config.id, socket.fileno()))

            self.dispatcher.schedule_task(
                Protocol.send_data, socket,
                MessageType.TRAINING_CONFIG, config.encode())

            # if number of clients is enough, start training
            self.num_clients_connected += 1
            if self.num_clients_connected == num_clients:
                self.event_all_clients_connected.set()

    def process_bye(self, socket, _, /, num_clients):
        # skip non-registered clients
        if socket not in self.clients_socket_id_map:
            return

        with self.clients_info_lock:
            client_id = self.clients_socket_id_map[socket]
            self.log.info("Connection to client {} closed (fd={})".format(
                socket.fileno(), client_id))
            # self.clients_socket_id_map.pop(socket)
            # self.clients_id_socket_map.pop(client_id)

            self.num_clients_connected -= 1

            # if all clients leaves while training already started, it means
            # a abnormal state. force shutdown. (normal shutdown should be
            # handled by subclass protocol)
            if self.event_all_clients_connected.is_set() and \
                    self.num_clients_connected == 0:
                self.log.info("All clients are gone, shutting down...")
                self.cleanup()
                # this function is called from dispatcher thread, so force
                # stopping the main main thread is necessary. see:
                # https://stackoverflow.com/questions/1489669/how-to-exit-the-entire-application-from-a-python-thread
                os._exit(1)

    def process_dispatcher_shutdown(self, num_clients, batch_size):
        self.log.info("Exception in server. Stopping ...")
        self.cleanup()
        os._exit(1)

    def sigint_handler(self, *_, **__):
        self.log.warning('SIGINT detected. Stopping everything...')
        self.cleanup()
        sys.exit(0)

    def __cleanup(self, args):
        """cleanup hooks"""
        self.cleanup_hook(args)
        # cleanup and close all sockets gracefully
        self.dispatcher.stop()

    def cleanup_hook(self, args):
        pass
