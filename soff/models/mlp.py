"""Simple MLP Model for the MNIST dataset"""
import torch
from torch import nn
from torch.nn import functional as F

import logging
log = logging.getLogger(__name__)


class MnistMlp(nn.Module):
    """Multi-laye perceptron for MNIST"""

    def __init__(self, _, dataset):
        super().__init__()
        self.linear1 = nn.Linear(784, 250)
        self.linear2 = nn.Linear(250, 100)
        self.linear3 = nn.Linear(300, dataset.num_classes())

    def forward(self, x):
        x = x.flatten(start_dim=2)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = x.flatten(start_dim=1)
        x = self.linear3(x)
        return F.log_softmax(x, dim=1)


class TurbineMlp(nn.Module):
    def __init__(self, cfg, dataset) -> None:
        super().__init__()
        num_features = len(dataset[0][0])
        hidden_dim = 100
        self.linear1 = nn.Linear(num_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.xavier_uniform_(self.linear4.weight)
        nn.init.xavier_uniform_(self.linear5.weight)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), negative_slope=0.00001)
        x = F.leaky_relu(self.linear2(x), negative_slope=0.00001)
        x = F.leaky_relu(self.linear3(x), negative_slope=0.00001)
        x = F.leaky_relu(self.linear4(x), negative_slope=0.00001)
        x =  torch.sigmoid(self.linear5(x).view(-1))
        return x
