import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
import math

from .data import load_iter
from .model import RNNLM
from .trainer import train, evaluate


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
if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')


model = RNNLM(input_dim, emb_dim, hid_dim, n_layers, dropout).to(device)

optimizer = optim.Adam(model.parameters())

pad_idx = TEXT.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

best_valid_loss = float('inf')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


for epoch in range(num_epochs):
    start_time = time.time()

    train_loss = train(model, train_iter, optimizer, criterion, clip)
    valid_loss = evaluate(model, valid_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f'| Epoch: {epoch+1:03} | Time: {epoch_mins}m {epoch_secs}s| \
        Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | \
        Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
