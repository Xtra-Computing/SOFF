import uuid
import unittest
from soff.security.aes import Key, KeyChain


class TestKey(unittest.TestCase):
    def test_encrypt(self):
        key = Key(True)

        random_data = uuid.uuid4().bytes + uuid.uuid4().bytes
        decrypted_data = key.decrypt(key.encrypt(random_data))
        self.assertEqual(random_data, decrypted_data)

        random_data = uuid.uuid4().bytes
        decrypted_data = key.decrypt(key.encrypt(random_data))
        self.assertEqual(random_data, decrypted_data)

        random_data = bytearray()
        for i in range(1000000):
            random_data.extend(uuid.uuid4().bytes)
        decrypted_data = key.decrypt(key.encrypt(random_data))
        self.assertEqual(random_data, decrypted_data)


class TestKeyRing(unittest.TestCase):
    def test_serialize(self):
        keychain = KeyChain()
        for _ in range(3):
            keychain.add_key(Key(gen_key=True))

        serialized_data = keychain.seralize_keys()
        deserialized_chain = KeyChain()
        deserialized_chain.deseralize_keys(serialized_data)

        for i, (orig_key, imported_key) in enumerate(
                zip(keychain, deserialized_chain)):
            for _ in range(3):
                random_data = uuid.uuid4().bytes
                self.assertEqual(
                    random_data,
                    orig_key.decrypt(imported_key.encrypt(random_data)),
                    "{}: encrypt - decrypt test failed".format(i))


if __name__ == "__main__":
    unittest.main()
