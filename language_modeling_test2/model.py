import torch
import torch.nn as nn
import numpy as np


class RNNLM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout=0):
        super(RNNLM, self).__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.linear = nn.Linear(hid_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        out, (hidden, cell) = self.rnn(embedded)
        out = out.reshape(-1, out.size(2))
        out = self.linear(out)

        return out, (hidden, cell)
