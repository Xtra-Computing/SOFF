"""Full-featured FL base client"""
from typing import List
import torch
from torch import Tensor
from munch import Munch
from .base_client import BaseClient
from ...security import aes
from ...security import rsa
from ...compressors.none import NoCompress
from ...communications.protocol import MessageType, SecureForward


class FFBaseClient(BaseClient):
    """Full-featured base client for federated learning"""

    def __init__(self, cfg: Munch):
        super().__init__(cfg)
        self.__init_secure_aggregation(cfg)

    def __init_secure_aggregation(self, cfg):
        if cfg.encryption.secure_aggregation_method is None:
            self.secure_aggregation = False
            return

        if cfg.encryption.secure_aggregation_method == 'SS':
            self.secure_aggregation = True
            self.secret_splitting_num = cfg.encryption.secret_split_num

            self.rsa_key = rsa.Key(gen_priv_key=True)
            self.rsa_keyring = rsa.KeyChain()

            # Announce RSA pubkey
            self.dispatcher.send_msg(
                MessageType.RSA_PUBKEY, self.rsa_key.seralize_pubkey())

            msg_type, data = self.dispatcher.recv_msg()
            assert msg_type == MessageType.RSA_PUBKEY
            self.rsa_keyring.deseralize_pubkeys(data)
            self.log.info("RSA Keychain received")
        else:
            raise NotImplementedError("Not implemented yet")

    def secure_exchange_splits(self, params: List[Tensor]) -> List[Tensor]:
        assert self.cfg.encryption.secure_aggregation_method == "SS", \
            "Aggregation method must be secret sharing"

        # TODO: Negotiate to ensure upper bound is larger than
        # 2 * max(|element in grad|). Currently we simply use a
        # large enough value
        UPPER_BOUND = 514229.0

        params_copy = [param.detach().float().clone() for param in params]
        random_split = [torch.zeros_like(param).float() for param in params]

        # offset all values to positive number
        for param in params_copy:
            param.add_(UPPER_BOUND / 2)
            assert (param > 0).all() and (param < UPPER_BOUND).all()

        # split grdient and sent to other clients
        for i in range(self.secret_splitting_num - 1):
            with torch.no_grad():
                for param, rand in zip(params_copy, random_split):
                    rand.copy_(
                        torch.rand_like(param).to(param.device) * UPPER_BOUND)
                    param.add_(UPPER_BOUND)
                    param.subtract_(rand)
                    param.fmod_(UPPER_BOUND)

            # pack and encrypt with random aes key
            data = NoCompress().compress(random_split)
            aes_key = aes.Key(gen_key=True)
            data = aes_key.encrypt(data)
            dst = (self.client_id + i + 1) % self.cfg.client_server.num_clients
            key = self.rsa_keyring[dst].encrypt(aes_key.seralize_key())

            # send data to other clients
            data = SecureForward().set_data({
                'src': self.client_id, 'dst': dst,
                'key': key, 'data': data}).encode()
            self.log.info(
                "Sending split to client %s (%s bytes)", dst, len(data))

            self.dispatcher.send_msg(MessageType.SECURE_FWD, data)

        # receive splits from other clients
        for i in range(self.secret_splitting_num - 1):
            msg_type, data = self.dispatcher.recv_msg()
            assert msg_type == MessageType.SECURE_FWD, msg_type
            msg = SecureForward().decode(data)
            self.log.info("Received split from client %s", msg.data.src)

            aes_key = aes.Key()
            aes_key.deseralize_key(self.rsa_key.decrypt(msg.data.key))
            split = NoCompress().decompress(
                bytearray(aes_key.decrypt(msg.data.data)))
            for param, spl in zip(params_copy, split):
                param.add_(spl.to(self.devices[0]))
                param.fmod_(UPPER_BOUND)

        # offset the values back to normal
        with torch.no_grad():
            for param in params_copy:
                param.subtract_(UPPER_BOUND / 2)

        return [cpy.type(ori.dtype) for cpy, ori in zip(params_copy, params)]
