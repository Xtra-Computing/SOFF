import pathlib
import logging
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from oarf.compressors.compress_utils import (
    Offset, pack_int, pack_raw_data, unpack_int, unpack_raw_data)

log = logging.getLogger(__name__)


class Key:
    def __init__(self, gen_priv_key: bool = False, priv_key_path: str = None):
        """
        | `gen_priv_key`  | `priv_key_path` | result                        |
        |:---------------:|:---------------:|:-----------------------------:|
        | True            | True            | new key generated and saved   |
        | True            | False           | new key generated             |
        | True            | True            | key loaded from path          |
        | False           | False           | do nothing                    |
        """
        self.priv_key = None
        self.pub_key = None

        # max size = (bytes(rsa) - 2 * bytes(hash) - 2),
        # currently hard-coded to 190 = 256 - 2 * 32 - 2
        self.max_encrypt_size = 190

        if gen_priv_key:
            self.priv_key = RSA.generate(2048)
            if priv_key_path is not None:
                path = pathlib.Path(priv_key_path)
                with open(path.as_posix(), 'w') as f:
                    f.write(self.priv_key.export_key().decode('utf-8'))
        elif priv_key_path is not None:
            path = pathlib.Path(priv_key_path)
            if path.is_file():
                self.priv_key = RSA.importKey(open(path.as_posix()).read())
            else:
                raise Exception("Failed to open file {}".format(path.as_posix))

        if self.priv_key is not None:
            self.pub_key = self.priv_key.publickey()

            # delegate encrypt/decrypt function
            self.cipher = PKCS1_OAEP.new(self.priv_key, hashAlgo=SHA256)
            self.decrypt = self.cipher.decrypt

    def encrypt(self, data: bytes):
        assert (len(data) <= self.max_encrypt_size)
        return self.cipher.encrypt(data)

    def has_privkey(self):
        return self.priv_key is not None

    def has_pubkey(self):
        return self.pub_key is not None

    def seralize_pubkey(self) -> bytearray:
        data = bytearray()
        data.extend(pack_raw_data(bytearray(self.pub_key.export_key())))
        return data

    def deseralize_pubkey(self, data, offset: Offset = None):
        offset = offset or Offset()
        self.pub_key = RSA.import_key(bytes(unpack_raw_data(data, offset)))
        self.cipher = PKCS1_OAEP.new(self.pub_key, hashAlgo=SHA256)
        # we cannot decrypt without a priv key, so not setting self.decrypt
        return self


class KeyChain(dict):
    def add_key(self, id: int,  key: Key):
        assert isinstance(key, Key)
        self[id] = key

    def seralize_pubkeys(self) -> bytearray:
        data = bytearray()
        data.extend(pack_int(len(self)))
        for id, key in self.items():
            data.extend(pack_int(id))
            data.extend(key.seralize_pubkey())

        return data

    def deseralize_pubkeys(self, data: bytearray, offset: Offset = None):
        offset = offset or Offset()
        num_keys = unpack_int(data, offset)
        for _ in range(num_keys):
            id = unpack_int(data, offset)
            new_key = Key().deseralize_pubkey(data, offset)
            self[id] = new_key
        return self
