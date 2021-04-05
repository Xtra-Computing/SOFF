from torch import nn, randn


class LSTM(nn.Module):
    def __init__(self, output_size, embedding_dim,
                 hidden_dim=200, num_layers=2, drop_prob=0.5, *_, **__):
        super().__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # self.embedding = nn.Embedding(vocabSize, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        batchSize = x.size(0)

        # embeds = self.embedding(x)
        # lstmOut, hidden = self.lstm(embeds, hidden)
        lstmOut, hidden = self.lstm(
            x, (randn(self.num_layers, batchSize, self.hidden_dim).cuda(),
                randn(self.num_layers, batchSize, self.hidden_dim).cuda()))
        lstmOut = lstmOut.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstmOut)
        out = self.fc(out)

        sigmoidOut = self.sig(out)
        sigmoidOut = sigmoidOut.view(batchSize, -1)
        sigmoidOut = sigmoidOut[:, -1]

        return sigmoidOut.double()
