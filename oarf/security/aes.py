import pathlib
import logging
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from oarf.compressors.compress_utils import (
    Offset, pack_int, pack_raw_data, unpack_int, unpack_raw_data)

log = logging.getLogger(__name__)


class Key:
    def __init__(self, gen_key: bool = False, key_path: str = None):
        """
        | `gen_key`       | `key_path`      | result                        |
        |:---------------:|:---------------:|:-----------------------------:|
        | True            | True            | new key generated and saved   |
        | True            | False           | new key generated             |
        | True            | True            | key loaded from path          |
        | False           | False           | do nothing                    |
        """
        self.key = None
        if gen_key:
            self.key = get_random_bytes(16)
            if key_path is not None:
                path = pathlib.Path(key_path)
                with open(path.as_posix(), 'w') as f:
                    f.write(self.key)
        elif key_path is not None:
            path = pathlib.Path(key_path)
            if path.is_file():
                self.key = open(path.as_posix()).read()
            else:
                raise Exception("Failed to open file {}".format(path.as_posix))

    def encrypt(self, data: bytes) -> bytearray:
        # Usage of EAX allows detection of unauthorized modification
        # Other modes that is not ECB could also be used (like GCM)
        cipher = AES.new(self.key, AES.MODE_EAX)
        cipertext, tag = cipher.encrypt_and_digest(data)

        data = bytearray()
        data.extend(pack_raw_data(cipher.nonce))
        data.extend(pack_raw_data(cipertext))
        data.extend(pack_raw_data(tag))

        return data

    def decrypt(self, data: bytes, offset: Offset = None) -> bytes:
        """
        throws ValueError exception when AES tag verification fails
        """
        offset = offset or Offset()

        nonce = unpack_raw_data(data, offset)
        cipertext = unpack_raw_data(data, offset)
        tag = unpack_raw_data(data, offset)

        cipher = AES.new(self.key, AES.MODE_EAX, nonce=nonce)
        plaintext = cipher.decrypt(cipertext)
        cipher.verify(tag)
        log.debug("The message is authentic")

        return plaintext

    def seralize_key(self) -> bytearray:
        data = bytearray()
        data.extend(pack_raw_data(self.key))
        return data

    def deseralize_key(self, data, offset:Offset = None):
        offset = offset or Offset()
        self.key = bytes(unpack_raw_data(data, offset))
        return self


class KeyChain(list):
    def add_key(self, key: Key):
        assert isinstance(key, Key)
        self.append(key)

    def seralize_keys(self) -> bytearray:
        data = bytearray()
        data.extend(pack_int(len(self)))
        for key in self:
            data.extend(key.seralize_key())

        return data

    def deseralize_keys(self, data: bytearray, offset: Offset = None):
        offset = offset or Offset()
        num_keys = unpack_int(data, offset)
        for _ in range(num_keys):
            new_key = Key()
            new_key.deseralize_key(data, offset)
            self.append(new_key)
        return self
