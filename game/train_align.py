import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummaryX import summary

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] =str(seed)
seed=0
set_seed(seed)

parser = argparse.ArgumentParser(description="Train a SplitNN with dcor loss")
parser.add_argument("--setting", default="two", type=str, help="Training VFL setting")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs to run for")
parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
parser.add_argument("--lr", default=1e-3, type=float, help="Starting learning rate")
args = parser.parse_args()

setting = args.setting
max_min_scaler = lambda x : (x-np.min(x)) / (np.max(x)-np.min(x))
align_data = pd.read_csv('align_all_data.csv')
align_data = align_data.drop(columns=['title'])
align_data['price'] = align_data[['price']].apply(max_min_scaler)
cols = ['appid', 'steamid', 'type', 'release_year', 'required_age', 'is_multiplayer']
if setting == 'two':
    align_data['score'] = align_data[['score']].apply(max_min_scaler)
    align_data['genre'] = align_data['genre'].fillna(align_data['genre'].mode()[0])
    cols += ['score_phrase', 'genre', 'editors_choice']
elif setting == 'one':
    align_data = align_data.drop(columns=['score_phrase', 'score', 'genre', 'editors_choice'])

# category to int
counts = []
for col in cols:
    cats = sorted(align_data[col].unique().tolist())
    cat2i = {cat: i for i, cat in enumerate(cats)}
    counts.append(len(cat2i))
    align_data[col] = align_data[col].transform(lambda cat: cat2i[cat])

print(counts)

batch_size = args.batch_size
trn_vld_df, tst_df = train_test_split(align_data, test_size=0.2, random_state=seed)
trn_df, vld_df = train_test_split(trn_vld_df, test_size=0.1, random_state=seed)
print(trn_df.shape, vld_df.shape, tst_df.shape)

if setting == 'two':
    x_trn = [trn_df.appid, trn_df.steamid, trn_df.type, trn_df.release_year, trn_df.required_age,
             trn_df.is_multiplayer, trn_df.score_phrase, trn_df.genre, trn_df.editors_choice,
             trn_df[['price', 'score']].astype('float32')]
    x_vld = [vld_df.appid, vld_df.steamid, vld_df.type, vld_df.release_year, vld_df.required_age,
             vld_df.is_multiplayer, vld_df.score_phrase, vld_df.genre, vld_df.editors_choice,
             vld_df[['price', 'score']].astype('float32')]
    x_tst = [tst_df.appid, tst_df.steamid, tst_df.type, tst_df.release_year, tst_df.required_age,
             tst_df.is_multiplayer, tst_df.score_phrase, tst_df.genre, tst_df.editors_choice,
             tst_df[['price', 'score']].astype('float32')]
elif setting == 'one':
    x_trn = [trn_df.appid, trn_df.steamid, trn_df.type, trn_df.release_year,
             trn_df.required_age, trn_df.is_multiplayer, trn_df[['price']].astype('float32')]
    x_vld = [vld_df.appid, vld_df.steamid, vld_df.type, vld_df.release_year,
             vld_df.required_age, vld_df.is_multiplayer, vld_df[['price']].astype('float32')]
    x_tst = [tst_df.appid, tst_df.steamid, tst_df.type, tst_df.release_year,
             tst_df.required_age, tst_df.is_multiplayer, tst_df[['price']].astype('float32')]

x_trn = [torch.from_numpy(col.to_numpy()) for col in x_trn]
x_vld = [torch.from_numpy(col.to_numpy()) for col in x_vld]
x_tst = [torch.from_numpy(col.to_numpy()) for col in x_tst]
y_trn = torch.from_numpy(trn_df.label.astype('float32').to_numpy())
y_vld = torch.from_numpy(vld_df.label.astype('float32').to_numpy())
y_tst = torch.from_numpy(tst_df.label.astype('float32').to_numpy())

trn_set = torch.utils.data.TensorDataset(*x_trn, y_trn)
vld_set = torch.utils.data.TensorDataset(*x_vld, y_vld)
tst_set = torch.utils.data.TensorDataset(*x_tst, y_tst)
print(len(trn_set), len(vld_set), len(tst_set))

trn_loader = torch.utils.data.DataLoader(trn_set, batch_size=batch_size, shuffle=True)
vld_loader = torch.utils.data.DataLoader(vld_set, batch_size=batch_size, shuffle=False)
tst_loader = torch.utils.data.DataLoader(tst_set, batch_size=batch_size, shuffle=False)
print(len(trn_loader), len(vld_loader), len(tst_loader))


class DLRM(nn.Module):
    def __init__(self, top_mlp_units, bot_mlp_units, emb_dim, counts, num_dense):
        super().__init__()
        num_cat = len(counts)
        self.num_cat = num_cat
        self.num_dense = num_dense
        self.emb_dim = emb_dim
        
        embs = [nn.Embedding(cnt, emb_dim) for cnt in counts]
        self.embs = nn.ModuleList(embs)
        
        bot_mlp = []
        prev = num_dense
        for units in bot_mlp_units:
            bot_mlp.append(nn.Linear(prev, units))
            bot_mlp.append(nn.ReLU())
            prev = units
        bot_mlp.append(nn.Linear(prev, emb_dim))
        self.bot_mlp = nn.Sequential(*bot_mlp)
        
        top_mlp = []
        prev = emb_dim * (num_cat + 1)
        for units in top_mlp_units:
            top_mlp.append(nn.Linear(prev, units))
            top_mlp.append(nn.ReLU())
            prev = units
        top_mlp.append(nn.Linear(prev, 1))
        top_mlp.append(nn.Sigmoid())
        self.top_mlp = nn.Sequential(*top_mlp)

    def forward(self, inputs):  # inputs: [1d categorical feature, ..., nd dense features]
        embs = []
        
        for i in range(self.num_cat):
            emb = self.embs[i](inputs[i])
            embs.append(emb)
            
        dense_emb = self.bot_mlp(inputs[-1])
        embs.append(dense_emb)
        
        out = torch.cat(embs, dim=1)
        out = self.top_mlp(out)
        out = torch.flatten(out)
            
        return out
    
    def _initialize_weights(self):  # same as keras
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.05, 0.05)

top_mlp_units = [256, 128, 64]
bot_mlp_units = []
emb_dim = 16
if setting == 'two':
    num_dense = 2
elif setting == 'one':
    num_dense = 1
lr = args.lr

model = DLRM(top_mlp_units, bot_mlp_units, emb_dim, counts, num_dense)
# summary(model, next(iter(trn_loader))[:-1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=2, verbose=True, threshold=1e-4)


def train():
    model.train()
    trn_loss = 0.0
    trn_total = 0
    trn_correct = 0
    for data in trn_loader:
        inputs = [x.to(device) for x in data[:-1]]
        labels = data[-1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        cnt = inputs[0].size(0)
        trn_total += cnt
        trn_loss += loss.item() * cnt
        
        predicted = (outputs.detach().cpu().numpy() > 0.5)
        trn_correct += np.sum(labels.detach().cpu().numpy() == predicted)

    trn_loss /= trn_total
    return trn_loss, 100 * trn_correct / trn_total

def test(data_loader, status):
    model.eval()
    vld_loss = 0.0
    vld_total = 0
    vld_correct = 0
    with torch.no_grad():
        for data in data_loader:
            inputs = [x.to(device) for x in data[:-1]]
            labels = data[-1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            cnt = inputs[0].size(0)
            vld_total += cnt
            vld_loss += loss.item() * cnt
            
            predicted = (outputs.detach().cpu().numpy() > 0.5)
            vld_correct += np.sum(labels.detach().cpu().numpy() == predicted)

    vld_loss /= vld_total
    if status == 'vld':
        scheduler.step(vld_loss)
    return vld_loss, 100 * vld_correct / vld_total

epochs = args.epochs
for epoch in range(epochs):
    start_t = time.time()
    trn_loss, trn_acc = train()
    vld_loss, vld_acc = test(vld_loader, 'vld')
    tst_loss, tst_acc = test(tst_loader, 'tst')
    end_t = time.time()
    print('Epoch %d trn_loss: %.4f trn_acc: %.2f vld_loss: %.4f vld_acc: %.2f tst_loss: %.4f tst_acc: %.2f Time: %d s' %
          (epoch, trn_loss, trn_acc, vld_loss, vld_acc, tst_loss, tst_acc, end_t-start_t))

