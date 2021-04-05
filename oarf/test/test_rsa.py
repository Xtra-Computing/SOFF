import uuid
import unittest
from oarf.security.rsa import Key, KeyChain


class TestKey(unittest.TestCase):
    def test_encrypt(self):
        key = Key(True)
        self.assertTrue(key.has_privkey())
        self.assertTrue(key.has_pubkey())

        random_data = uuid.uuid4().bytes
        decrypted_data = key.decrypt(key.encrypt(random_data))
        self.assertEqual(random_data, decrypted_data)

        random_data = uuid.uuid4().bytes + uuid.uuid4().bytes
        decrypted_data = key.decrypt(key.encrypt(random_data))
        self.assertEqual(random_data, decrypted_data)


class TestKeyRing(unittest.TestCase):
    def test_serialize(self):
        keychain = KeyChain()
        for i in range(3):
            keychain.add_key(i, Key(gen_priv_key=True))

        serialized_data = keychain.seralize_pubkeys()
        deserialized_chain = KeyChain()
        deserialized_chain.deseralize_pubkeys(serialized_data)

        for i, (orig_key, imported_key) in enumerate(
                zip(keychain.values(), deserialized_chain.values())):
            random_data = uuid.uuid4().bytes
            self.assertEqual(
                random_data,
                orig_key.decrypt(imported_key.encrypt(random_data)),
                "{}: encrypt - decrypt test failed".format(i))


if __name__ == "__main__":
    unittest.main()
