import os
import re

import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from models.rnn import GRU
from utils.data import GRUDataset
from utils.loops.gru_loops import train_epoch, train, eval_epoch
from utils.preprocessing import extract_vocab, encode

DIR_PATH = 'data/'
SAVE_DIR = "saves/"
RANDOM_STATE = 42

data = pd.read_excel(f"{DIR_PATH}/19_35000.xlsx")

strcutures = data['SMILES'].tolist()
target = data['IC50'].tolist()

train_structures, val_structures, train_targets, val_targets = train_test_split(
    strcutures, target, test_size=0.1, shuffle=True, random_state=RANDOM_STATE
)

vocab, token2id, id2token = extract_vocab(train_structures)

mean = torch.tensor(train_targets).mean().item()
std = torch.tensor(train_targets).std().item()

train_structures = list(map(lambda x: encode(x, vocab, token2id), train_structures))
val_structures = list(map(lambda x: encode(x, vocab, token2id), val_structures))

max_length = max([len(struct) for struct in train_structures])

train_dataset = GRUDataset(train_structures, train_targets, max_length, token2id, mean, std)
val_dataset = GRUDataset(val_structures, val_targets, max_length, token2id, mean, std)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

model = GRU(
    num_embeddings=len(vocab),
    embedding_dim=64,
    hidden_size=64
)

epochs = 100
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.L1Loss()
metric_fn = r2_score
device = 'mps'

train(
    epochs, 
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn, 
    metric_fn,
    device,
    save_dir=SAVE_DIR,
    mean=mean,
    std=std
)