"""Simple CNN Model for the FEMNIST dataset"""

from torch import nn

class FemnistCnn(nn.Module):
    def __init__(self, _, dataset) -> None:
        super().__init__()
        self.layer_1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.layer_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)
        self.dropout_1 = nn.Dropout2d(p=0.5)
        self.layer_3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.layer_4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2)
        self.dropout_2 = nn.Dropout2d(p=0.5)
        self.linear = nn.Linear(512, dataset.num_classes())
        self.dropout_3 = nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU()

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

