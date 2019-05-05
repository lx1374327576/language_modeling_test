import torch
import torch.nn as nn

import os
import math

from .data import load_iter
from .model import RNNLM
from .trainer import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, valid_iter, test_iter, TEXT = load_iter()

input_dim = len(TEXT.vocab)
emb_dim = 256
hid_dim = 512
n_layers = 2
dropout = 0.5
num_epochs = 10
clip = 1

SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'lm_model.pt')

model = RNNLM(input_dim, emb_dim, hid_dim, n_layers, dropout).to(device)

model.load_state_dict(torch.load(MODEL_SAVE_PATH))

pad_idx = TEXT.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

test_loss = evaluate(model, test_iter, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
