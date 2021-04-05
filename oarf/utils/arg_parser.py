import argparse
from oarf.datasets.datasets import dataset_name_to_class


class ServerArgParser:
    server_only_args = [
        'log_file', 'log_level', 'tensorboard_log_dir', 'num_cache',
        'client_fraction', 'rate_limit']
    """A list of server-only argument (should not be passed to clients)"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Server-only arguments
        server_args = self.parser.add_argument_group(
            "\033[1mLog and Cache Config (S Only)\033[0m")
        server_args.add_argument(
            '-lf', '--log-file', default=None, type=str,
            help="Path to the logging file")
        server_args.add_argument(
            '-ll', '--log-level', default='info', type=str,
            choices=['debug', 'info', 'warning', 'error', 'critical'],
            help="Set log levels")
        server_args.add_argument(
            '-tl', '--tensorboard-log-dir', default=None, type=str,
            help="Tensorflow log directory")
        server_args.add_argument(
            '-nc', '--num-cache', default=5, type=int,
            help="# of received client models that exist at the same time\n"
            " in memory, 2 * nc should be less than total host memory size")
        server_args.add_argument(
            '-rl', '--rate-limit', default=0, type=int,
            help="send/recv rate limit, in Bytes/sec. 0 for no limit")

        # Basic configs for both clients and servers
        shared_args = self.parser.add_argument_group(
            "\033[1mTraining and Communication Setup (C/S Shared)\033[0m")
        shared_args.add_argument(
            '-n', '--num-clients', default=4, type=int, required=True,
            help="Number of clients")
        shared_args.add_argument(
            '-s', '--socket-type', default='unix', help='<unix|tcp>')
        shared_args.add_argument(
            '-ad', '--address', default='/tmp/fedavg-comm.sock', type=str,
            help="Socket address (unix socket only for local simulation, "
            "for cross-machine support, use tcp socket)")
        shared_args.add_argument(
            '-e', '--epochs', default=300, type=int,
            help="Number of epochs to train")
        shared_args.add_argument(
            '-nt', '--num-threads', default=5, type=int,
            help="Number of thread for sending model and receiving gradient.\n"
            " Should not exceed 2 * num-cache, otherwise waste resources.")
        shared_args.add_argument(
            '-sd', '--seed', type=int, default=0,
            help="random initial seed")
        shared_args.add_argument(
            '-le', '--log-every', type=int, default=1,
            help="Log (print/tensorboard) every X communication round.")
        shared_args.add_argument(
            '-lr', '--learning-rate', default=0.1, type=float,
            help="Initial lr. Also the target lr rate of warmup epochs.")
        shared_args.add_argument(
            '-lrf', '--lr-factor', type=float, default=0.1,
            help="Learning rate reduce factor.")
        shared_args.add_argument(
            '-bt', '--broadcast-type', default='gradient',
            choices=['gradient', 'model'],
            help="Server broadcast type. 'gradient' saves more communication, "
            "since it may benefit from server gradient compressor, "
            "but can only be used when client-fraction = 1 (e.g. fedsgd), "
            "since we need to ensure that global models are in sync.")

        shared_args.add_argument(
            '-d', '--datasets', default=['cifar10'], nargs='+',
            choices=dataset_name_to_class.keys(),
            help="Selected datasets must have the same prefix")
        shared_args.add_argument(
            '-td', '--test-datasets', default=None, nargs='+',
            help="Seleted test datasets must have the same prefix, "
            "Combined test dataset can be separted with ',' (without space), "
            "e.g. 'sentiment:amazon,sentiment:imdb'.\n"
            "Will use the test set tied to the train set if set to None.")
        shared_args.add_argument(
            '-ds', '--data-splitting', default='iid', type=str,
            choices=['iid', 'label-skew-dirichlet',
                     'quantity-skew-dirichlet', 'quantity-skew-powerlaw',
                     'realistic', 'realistic-subsample'],
            help="Method to split the datasets. If multiple datasets are "
            "selected, then those datasets are first combined into a large "
            "dataset before splitting. Exceptions are the 'realistic' and "
            "'realistic-subsample' method, "
            "which is only applicable to non-synthetic datset, and if it is "
            "selected, --num-clients (-n) must match the number of dataset")
        shared_args.add_argument(
            '-al', '--alpha', default=0.5, type=float,
            help="Alpha value for skew partitioning")
        shared_args.add_argument(
            '-M', '--model', default='ResNet18', type=str,
            choices=[
                'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                'LSTM', 'VGG16'],  # TODO
            help="Underlying model.")
        shared_args.add_argument(
            '-o', '--optimizer', default='SGD', type=str,
            choices=['SGD', 'Adam'],
            help="Optimizer for local trainig")

        fednova_args = self.parser.add_argument_group(
            "\033[1mFedNova-related Arguments (S Only)\033[0m")
        fednova_args.add_argument(
            '-cf', '--client-fraction', default=0.2, type=float,
            help="Fraction of clients selected each communication round.")
        fednova_args.add_argument(
            '-gc', '--gradient-correction', action='store_true',
            help="Enable gradient correction")

        fedprox_args = self.parser.add_argument_group(
            "\033[1mFedProx-related Arguments (C/S Shared)\033[0m")
        fedprox_args.add_argument(
            '-mu', '--mu', default=0.1, type=float,
            help="FedProx parameter μ")

        # Client-only training config
        client_args = self.parser.add_argument_group(
            "\033[1mTrainig Config (C Only)\033[0m")
        client_args.add_argument(
            '-ap', '--average-policy', default='epoch', type=str,
            choices=['epoch', 'iter'],
            help="Average per X epochs/iters.")
        client_args.add_argument(
            '-a', '--average-every', default=1, type=int,
            help="Average interval (Number of local epochs/iters)")
        client_args.add_argument(
            '-bs', '--batch-size', default=128, type=int, help="Batch size")
        client_args.add_argument(
            '-we', '--warmup-epochs', default=10, type=int,
            help="Number of epochs for warmup training")
        client_args.add_argument(
            '-wd', '--weight-decay', default=5e-4, type=float,
            help="Weight decay")
        client_args.add_argument(
            '-rd', '--rotation-degree', default=5.0, type=float,
            help="Random rotation degree ranges, for data augmentation")
        client_args.add_argument(
            '-m', '--momentum', default=0.9, type=float, help="SGD momentum")
        client_args.add_argument(
            '-p', '--patience', default=10, type=int, help="Plateau patience")
        # client_args.add_argument(
        #     '-sf', '--similarity-shuffle', action='store_true',
        #     help="Experimental")
        client_args.add_argument(
            '-br', '--batchnorm-runstat', action='store_true',
            help="Track the running statistics of batchnorm layer")

        enc_args = self.parser.add_argument_group(
            "\033[1mEncryption-related Arguments(C/S Shared)\033[0m")
        enc_args.add_argument(
            '-sam', '--secure-aggregation-method', default=None,
            choices=[None, 'SS', 'HE', 'SMC'],
            help="Method for secure aggregation (currently only SS supported)"
            "\n  SS: Secret Sharing"
            "\n  HE: Homomorphic Encryption"
            "\n  SMC: Secure Multiparty Computation")
        enc_args.add_argument(
            '-ssn', '--secret_splitting_num', default=2, type=int,
            help="Number of splits when using the SS aggregation method.")

        # Client-server shared Compression-related args
        compress_args = self.parser.add_argument_group(
            "\033[1mCompression-related Arguments (C/S Shared)\033[0m")

        compress_args.add_argument(
            '-cc', '--clientside-compressor', default='none', type=str,
            choices=['none', 'topk_permodel', 'topk_perlayer',
                     'randk_permodel', 'randk_perlayer',
                     'rankk_perlayer', 'svd_perlayer', 'ada_rankk_perlayer'],
            help="Compressor for compressing gradient.")
        compress_args.add_argument(
            '-sc', '--serverside-compressor', default='none', type=str,
            choices=['none', 'sparse', 'topk_permodel', 'topk_perlayer'],
            help="Compressor for compressing model.")
        compress_args.add_argument(
            '-sr', '--server-ratio', default=0.25, type=float,
            help="Subsample ratio @server when using top/rand compressor")
        compress_args.add_argument(
            '-cr', '--client-ratio', default=0.01, type=float,
            help="Subsample ratio @client when using top/rand compressor")
        compress_args.add_argument(
            '-sk', '--server-rank', default=3, type=int,
            help="The matrix rank @server when using rank compressor")
        compress_args.add_argument(
            '-ck', '--client-rank', default=3, type=int,
            help="The matrix rank @client when using rank compressor")

        # Client-only compression-related args
        c_compress_args = self.parser.add_argument_group(
            "\033[1mCompression-related Arguments (C Only)\033[0m")

        c_compress_args.add_argument(
            '-gm', '--global-momentum', default=0.0, type=float,
            help="Set global momentum value "
            "(inspired by https://openreview.net/forum?id=SkhQHMW0W)")
        c_compress_args.add_argument(
            '-gl', '--global-learning-rate', default=1.0, type=float,
            help="Global learning rate, used together with global momentum")
        c_compress_args.add_argument(
            '-mm', '--momentum-masking', action='store_true',
            help="Use momentum masking to filter the global momentum")
        c_compress_args.add_argument(
            '-smm', '--server-momentum-masking', action='store_true',
            help="Use momentum masking to filter the global momentum, "
            "but use server gradient as a reference")
        c_compress_args.add_argument(
            '-ef', '--error-feedback', action='store_true',
            help="Use error feedback to correct the gradient.")
        c_compress_args.add_argument(
            '-ed', '--error-decay', default=0.0, type=float,
            help="(Experimental) reduce size of error feedback"
        )

        # TODO: quantization
        c_quantize_args = self.parser.add_argument_group(
            "\033[1mQuantization-related Arguments (C Only)\033[0m")

        c_quantize_args.add_argument(
            '-q', '--quantize', action='store_true',
            help="Use quantization on gradients")
        c_quantize_args.add_argument(
            '-qb', '--quantize-bits', default=2, type=int,
            help="Quantization bits")

        c_dp_args = self.parser.add_argument_group(
            "\033[1mDP-related Arguments (C Only)\033[0m")
        c_dp_args.add_argument(
            '-dt', '--dp-type', default=None, choices=[None, 'rdp'],
            help="Type of differential privacy to use")
        c_dp_args.add_argument(
            '-ep', '--epsilon', default=2.0, type=float,
            help="ε value for rdp")
        c_dp_args.add_argument(
            '-de', '--delta', default=0, type=float,
            help="δ value for rdp, set to 0 for automatic setup")
        c_dp_args.add_argument(
            '-cl', '--clip', default=5.0, type=float,
            help="Gradient clipping value for DP")

    def parse_args(self):
        args = self.parser.parse_args()
        assert 1 <= args.quantize_bits <= 32
        return args

    @ classmethod
    def get_args_for_clients(cls, args: dict) -> dict:
        return {k: v for (k, v) in args.items()
                if k not in cls.server_only_args}


class ClientArgParser:
    """client-only arguments"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument(
            '-lf', '--log-file', default=None, type=str,
            help="Path to the log file")
        self.parser.add_argument(
            '-ll', '--log-level', default='info', type=str,
            choices=['debug', 'info', 'warning', 'error', 'critical'],
            help="Set log levels")
        self.parser.add_argument(
            '-tl', '--tensorboard-log-dir', default=None, type=str,
            help="Tensorflow log directory")

        self.parser.add_argument(
            '-s', '--socket-type', default='unix', help='<unix|tcp>')
        self.parser.add_argument(
            '-ad', '--address', default='/tmp/fedavg-comm.sock', type=str,
            help="Socket address (unix socket only for local simulation,"
            "for cross-machine support, use tcp socket)")

        fednova_args = self.parser.add_argument_group(
            "\033[1mFedNova-related Arguments (S Only)\033[0m")

        fednova_args.add_argument(
            '-sw', '--step-weights', default=1,
            help="When given a number, local steps has equal weights ( "
            "equivalent to fedavg). When given a list, each local step has "
            "weight corresponding to the value in the list, and the size of "
            "the list must match the number of local steps. "
            "Note that when used together with SGD momentum, the momentum "
            "factor will be combined into this argument before being sent to "
            "the server for aggregation.")

        self.parser.add_argument(
            '-sa', '--svd-analysis', action='store_true',
            help="Log svd analysis info (can be very slow)")

    def parse_args(self):
        args = self.parser.parse_args()
        return args
