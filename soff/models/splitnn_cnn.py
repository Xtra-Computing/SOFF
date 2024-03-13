"""Simple SplitNN CNN Module"""
from itertools import chain
from torch.nn import functional as F
from torch import nn
from soff.utils.arg_parser import ArgParseOption, options


@options(
    "SplitNN Server Model Configs",
    ArgParseOption(
        'spcnn.i', 'spcnn.input-dim', type=int, metavar='DIM',
        help='input dimension size'),
    ArgParseOption(
        'spcnn.sh', 'spcnn.server-hidden-dim', default=512, type=int, metavar='DIM',
        help='hidden dimension size'))
class SplitnnBinaryCnnServer(nn.Module):
    def __init__(self, cfg, dataset) -> None:
        super().__init__()
        assert cfg.model.spcnn.input_dim is not None
        self.linear1 = nn.Linear(
            cfg.model.spcnn.input_dim, cfg.model.spcnn.server_hidden_dim * 2)
        self.conv1 = nn.Conv1d(1, 1, 3, 2, 1)
        self.convs1 = nn.Sequential(
            nn.Conv1d(1, 1, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.001),
            nn.Conv1d(1, 1, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.001),
        )
        self.convs2 = nn.Sequential(
            nn.Conv1d(1, 1, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.001),
            nn.Conv1d(1, 1, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.001),
        )
        self.linear2 = nn.Linear(cfg.model.spcnn.server_hidden_dim, 1)

        nn.init.xavier_uniform_(
            self.linear1.weight, nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(
            self.conv1.weight, nn.init.calculate_gain('leaky_relu'))
        for l in chain(self.convs1.modules(), self.convs2.modules()):
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)
        nn.init.xavier_uniform_(
            self.linear2.weight, nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), negative_slope=0.001)
        x = x.view(x.shape[0], 1, -1)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.001)
        x_2 = self.convs1(x)
        x_3 = self.convs2(x + x_2)
        x = self.linear2(x_2 + x_3)
        return x.flatten()


@options(
    "SplitNN Client Model Configs",
    ArgParseOption(
        'spcnn.o', 'spcnn.output-dim', type=int, metavar='DIM',
        help='number of output dimensions'),
    ArgParseOption(
        'spcnn.ch', 'spcnn.client-hidden-dim', default=512, type=int, metavar='DIM',
        help='hidden dimension size'))
class SplitnnBinaryCnnClient(nn.Module):
    def __init__(self, cfg, dataset) -> None:
        super().__init__()
        assert cfg.model.spcnn.output_dim is not None
        num_features = len(dataset[0][0])
        self.linear1 = nn.Linear(
            num_features, cfg.model.spcnn.client_hidden_dim * 2)
        self.conv1 = nn.Conv1d(1, 1, 3, 2, 1)
        self.convs1 = nn.Sequential(
            nn.Conv1d(1, 1, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.001),
            nn.Conv1d(1, 1, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.001),
        )
        self.convs2 = nn.Sequential(
            nn.Conv1d(1, 1, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.001),
            nn.Conv1d(1, 1, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.001),
        )
        self.linear2 = nn.Linear(
            cfg.model.spcnn.client_hidden_dim, cfg.model.spcnn.output_dim)

        # Initialize layers
        nn.init.xavier_uniform_(
            self.linear1.weight, nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(
            self.conv1.weight, nn.init.calculate_gain('leaky_relu'))
        for l in chain(self.convs1.modules(), self.convs2.modules()):
            if isinstance(l, nn.Conv1d):
                nn.init.xavier_uniform_(l.weight)
        nn.init.xavier_uniform_(
            self.linear2.weight, nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), negative_slope=0.001)
        x = x.view(x.shape[0], 1, -1)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.001)
        x_2 = self.convs1(x)
        x_3 = self.convs2(x + x_2)
        x = F.leaky_relu(self.linear2(x_2 + x_3), negative_slope=0.001)
        return x.view(x.shape[0], -1)
