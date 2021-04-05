import json
from oarf.metrics.analyzer import ClientAnalyzer
from torch.utils.data import DataLoader
from oarf.compressors.compressors import Compressor
from oarf.utils.arg_parser import ClientArgParser
from oarf.communications.dispatcher import ClientDispatcher
from oarf.communications.protocol import MessageType, TrainingConfig
from oarf.privacy.rdp import compute_gaussian_sigma
from oarf.utils.logging import Logger
from oarf.utils.training import Deterministic
from oarf.datasets.datasets import create_dataset
from oarf.security import rsa


class BaseClient(Logger, Deterministic, Compressor):
    def __init__(self, **kwargs):
        self.init_dispatcher(**kwargs)

        # send handshake (register) message
        self.dispatcher.send_msg(MessageType.HANDSHAKE, b'Hello!')

        # receive and merge config
        msg_type, data = self.dispatcher.recv_msg()
        assert (msg_type == MessageType.TRAINING_CONFIG)
        server_args = TrainingConfig().decode(data)
        self.args = {**server_args.__dict__, **kwargs}

        # use combined config to initialize everything else
        super().__init__(**self.args, logger_name=self.__class__.__name__)
        self.log.info('\n' + json.dumps(self.args, indent=2))

        self.init_dataset(**self.args)
        self.init_dp(**self.args)

        self.init_secure_aggregation(**self.args)
        self.analyzer = ClientAnalyzer()

    @classmethod
    def start(cls):
        argparser = ClientArgParser()
        args = argparser.parse_args()

        client = cls(**args.__dict__)

        # client args has higher priority
        client.init_training(**client.args)
        client.start_training(**client.args)
        client.cleanup()

    # TODO: merge into __init__
    def init_training(self, *_, **__):
        raise NotImplementedError

    def start_training(self, *_, **__):
        raise NotImplementedError

    def init_dispatcher(self, socket_type, address, **_):
        self.dispatcher = ClientDispatcher(socket_type, address)
        self.dispatcher.start()

    # Initialize dataset ######################################################
    def init_dataset(self, datasets, test_datasets, data_splitting, alpha,
                     num_clients, id, batch_size, seed,
                     rotation_degree, **_):

        # we leave eval to server, only using train/test here
        self.train_dataset = create_dataset(
            datasets=datasets, data_splitting=data_splitting,
            mode='train', rotation_degree=rotation_degree,
            num_clients=num_clients, client_id=id, alpha=alpha)
        # self.test_dataset = create_dataset(
        #     datasets=datasets, data_splitting=data_splitting,
        #     mode='test', rotation_degree=rotation_degree,
        #     num_clients=num_clients, client_id=id, alpha=alpha)
        test_datasets = ([','.join(datasets)] if test_datasets is None
                         else test_datasets)
        self.test_datasets = {
            tdss: create_dataset(
                datasets=tdss.split(','), data_splitting='iid',
                mode='test', num_clients=1, client_id=None)
            for tdss in test_datasets
        }

        self.train_loader = DataLoader(
            self.train_dataset, batch_size, shuffle=False)
        # self.test_loader = DataLoader(
        #     self.test_dataset, batch_size, shuffle=False)
        self.test_loaders = {
            k: DataLoader(v, batch_size, shuffle=False)
            for k, v in self.test_datasets.items()
        }

    def init_secure_aggregation(
            self, secure_aggregation_method, secret_splitting_num, *_, **__):
        if secure_aggregation_method is None:
            self.secure_aggregation = False
            return
        elif secure_aggregation_method == 'SS':
            self.secure_aggregation = True
            self.secret_splitting_num = secret_splitting_num

            self.rsa_key = rsa.Key(gen_priv_key=True)
            self.rsa_keyring = rsa.KeyChain()

            # announce RSA pubkey
            self.dispatcher.send_msg(
                MessageType.RSA_PUBKEY, self.rsa_key.seralize_pubkey())

            msg_type, data = self.dispatcher.recv_msg()
            assert(msg_type == MessageType.RSA_PUBKEY)
            self.rsa_keyring.deseralize_pubkeys(data)
            self.log.info("RSA Keychain received")
        else:
            raise NotImplementedError("Not implemented yet")

    def init_dp(self, dp_type, epsilon, delta, batch_size, epochs, **_):
        self.dp_type = dp_type
        self.dp_noise_level = 0
        if dp_type == 'rdp':
            computed_delta = (min(1e-5, 1. / len(self.train_dataset))
                              if delta == 0 else delta)
            self.dp_noise_level = compute_gaussian_sigma(
                epsilon, computed_delta,
                batch_size, len(self.train_dataset), epochs)

    def cleanup(self, **_):
        pass
    # TODO: Model saving utilities
