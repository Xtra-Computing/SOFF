import unittest
from oarf.privacy.rdp import compute_gaussian_sigma


class TestRDP(unittest.TestCase):
    def test_compute_sigma(self):
        print(compute_gaussian_sigma(2.0, 1e-5, 128, 100000, 300))


if __name__ == "__main__":
    unittest.main()
