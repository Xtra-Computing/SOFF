"""Simple SplitNN CNN Module"""
from itertools import chain
from torch.nn import functional as F
from torch import nn
from ..utils.arg_parser import ArgParseOption, options


@options(
    "SplitNN Server Model Configs",
    ArgParseOption(
        'spnn.i', 'spnn.input-dim', type=int, metavar='DIM',
        help='input dimension size'),
    ArgParseOption(
        'spnn.sh', 'spnn.server-hidden-dim', default=512, type=int, metavar='DIM',
        help='hidden dimension size'))
class SplitnnBinaryMlpServer(nn.Module):
    def __init__(self, cfg, dataset) -> None:
        super().__init__()
        assert cfg.model.spnn.input_dim is not None
        self.linear1 = nn.Linear(
            cfg.model.spnn.input_dim, cfg.model.spnn.server_hidden_dim)
        self.linear2 = nn.Linear(cfg.model.spnn.server_hidden_dim, 1)

        nn.init.xavier_uniform_(
            self.linear1.weight, nn.init.calculate_gain('leaky_relu'))
        # for l in chain(self.linears1.modules(), self.linears2.modules()):
        #     if isinstance(l, nn.Linear):
        #         nn.init.xavier_uniform_(l.weight)
        nn.init.xavier_uniform_(
            self.linear2.weight, nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), negative_slope=0.0001)
        x = self.linear2(x)
        return x.flatten()


@options(
    "SplitNN Client Model Configs",
    ArgParseOption(
        'spnn.o', 'spnn.output-dim', type=int, metavar='DIM',
        help='number of output dimensions'),
    ArgParseOption(
        'spnn.ch', 'spnn.client-hidden-dim', default=512, type=int, metavar='DIM',
        help='hidden dimension size'))
class SplitnnBinaryMlpClient(nn.Module):
    def __init__(self, cfg, dataset) -> None:
        super().__init__()
        assert cfg.model.spnn.output_dim is not None
        num_features = len(dataset[0][0])
        self.linear1 = nn.Linear(
            num_features, cfg.model.spnn.client_hidden_dim)
        self.linear2 = nn.Linear(
            cfg.model.spnn.client_hidden_dim, cfg.model.spnn.output_dim)

        nn.init.xavier_uniform_(
            self.linear1.weight, nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(
            self.linear2.weight, nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), negative_slope=0.0001)
        x = F.leaky_relu(self.linear2(x), negative_slope=0.0001)
        return x
