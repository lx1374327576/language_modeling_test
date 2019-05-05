import torch

from torchtext.datasets import PennTreebank
from torchtext.data import Field, BPTTIterator

import spacy

import random


def load_iter():
    SEED = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random.seed(SEED)
    torch.manual_seed(SEED)

    spacy_en = spacy.load('en')

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings (tokens)
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

    train_set, valid_set, test_set = PennTreebank.splits(TEXT)

    TEXT.build_vocab(train_set)

    train_iter, valid_iter, test_iter = BPTTIterator.splits(
        (train_set, valid_set, test_set),
        batch_size=64, bptt_len=6, device=device)

    return train_iter, valid_iter, test_iter, TEXT
