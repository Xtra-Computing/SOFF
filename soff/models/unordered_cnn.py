"""Simple CNN Model for the FEMNIST dataset"""
from torch import nn
from munch import Munch
from ..utils.arg_parser import ArgParseOption, options
from .unordered_net import _UnorderedNet, _MatchableConv2d, _MatchableLinear


@options(
    "Unordered CNN (for FeMNIST) Model Configs",
    ArgParseOption(
        'ufcnn.dp', 'unordered-femnist-cnn.drop-prob', type=float, default=0.5,
        help="Dropout probability"))
class UnorderedFemnistCnn(_UnorderedNet):
    def __init__(self, cfg: Munch, dataset) -> None:
        super().__init__(cfg, dataset)

    def init_submodules(self, cfg, dataset):
        drop_prob = cfg.model.unordered_femnist_cnn.drop_prob
        self.layer_1 = _MatchableConv2d(
            3, 32, kernel_size=3, stride=2, padding=1)
        self.layer_2 = _MatchableConv2d(
            32, 64, kernel_size=3, stride=2, padding=1)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)
        self.dropout_1 = nn.Dropout2d(p=drop_prob)
        self.layer_3 = _MatchableConv2d(
            64, 64, kernel_size=3, stride=2, padding=1)
        self.layer_4 = _MatchableConv2d(
            64, 128, kernel_size=3, stride=2, padding=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2)
        self.dropout_2 = nn.Dropout2d(p=drop_prob)
        self.linear = _MatchableLinear(512, dataset.num_classes())
        self.dropout_3 = nn.Dropout2d(p=drop_prob)
        self.relu = nn.ReLU()

    def init_matchable_layers(self):
        self.matchable_layers = [
            [(name, module)] for name, module in self.named_modules()
            if isinstance(module, (nn.Linear, nn.Conv2d))]

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.maxpool_1(x)
        x = self.dropout_1(x)

        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.maxpool_2(x)
        x = self.dropout_2(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.linear(x))
        x = self.dropout_3(x)
        return x
