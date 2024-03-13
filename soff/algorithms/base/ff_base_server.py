"""Full featured base server and config parser"""
import threading
from functools import partial

from .base_server import BaseServerConfParser, BaseServer
from .base_server_scheduler import StaticBaseServerScheduler
from ...security import rsa
from ...utils.arg_parser import  EncryptionConfParser
from ...privacy.rdp import DPConfParser
from ...communications.protocol import Protocol, MessageType, SecureForward


class FFBaseServerConfParser(
        DPConfParser, EncryptionConfParser, BaseServerConfParser):
    """Parse base configs that will used by the full-featured server."""


class FFBaseServer(BaseServer):
    """
    Full featured base server. Includes extra infrastructures for features
    like as compression and secure aggregation.
    """

    def __init__(self, cfg, scheduler=StaticBaseServerScheduler):
        super().__init__(cfg, scheduler)
        self.__init_secure_aggregation(cfg)

    def __init_secure_aggregation(self, cfg):
        if cfg.encryption.secure_aggregation_method is None:
            self.secure_aggregation = False
            return

        num_clients = self.cfg.client_server.num_clients
        assert cfg.encryption.secure_aggregation_method == 'SS'
        assert 2 <= cfg.encryption.secret_split_num <= num_clients

        self.log.info(
            "Waiting for %s clients' pubkeys ...", cfg.client_server.num_clients)

        self.secret_splitting_num = cfg.encryption.secret_split_num
        self.secure_aggregation = True
        self.rsa_keyring_lock = threading.Lock()
        self.rsa_keyring = rsa.KeyChain()
        self.scheduler.dispatcher.register_msg_event(
            MessageType.RSA_PUBKEY,
            partial(self.register_client_pubkey, num_clients=num_clients))
        self.scheduler.dispatcher.register_msg_event(
            MessageType.SECURE_FWD, self.forward_grad_split)

    def register_client_pubkey(self, socket, data, /, num_clients):
        """
        Register client publickey to keyring and broadcast the keyring if
        all clients' public keys are collected.
        """
        with self.rsa_keyring_lock:
            cli_id = self.scheduler.clients_socket_id_map[socket]
            self.rsa_keyring.add_key(cli_id, rsa.Key().deseralize_pubkey(data))
            self.log.info("Pubkey of client %s ✔", cli_id)

            # broadcast all pubkey if pubkey of all clients are received
            if len(self.rsa_keyring) == num_clients:
                self.log.info("Broadcasting public keychain ...")
                self.scheduler.dispatcher.schedule_broadcast(
                    MessageType.RSA_PUBKEY,
                    data=self.rsa_keyring.seralize_pubkeys(),
                    datalogger=self.datalogger)

    def forward_grad_split(self, _, data):
        """simply forward msg to required destination"""
        msg = SecureForward().decode(data)
        self.log.info(
            "Forwarding message %s -> %s", msg.data.src, msg.data.dst)
        self.scheduler.dispatcher.schedule_task(
            Protocol.send_data,
            self.scheduler.clients_id_socket_map[msg.data.dst],
            MessageType.SECURE_FWD, data)
