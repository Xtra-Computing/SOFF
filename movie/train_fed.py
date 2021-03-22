import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # uncomment to get reproducible results
seed = 0

import numpy as np
import torch
import random
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    torch.backends.cudnn.benchmark = False
set_seed(seed)

import pandas as pd
import numpy as np
import os
from time import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# # Preprocess MovieLens-1M
# MOVIELENS_DIR = 'ml-1m'
# USER_DATA_FILE = 'users.dat'
# MOVIE_DATA_FILE = 'movies.dat'
# RATING_DATA_FILE = 'ratings.dat'

# AGES = { 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
# OCCUPATIONS = { 0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
#                 4: "college/grad student", 5: "customer service", 6: "doctor/health care",
#                 7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
#                 12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
#                 17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer" }

# ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE), 
#                     sep='::', 
#                     engine='python', 
#                     encoding='latin-1',
#                     names=['user_id', 'movie_id', 'rating', 'timestamp'])

# users = pd.read_csv(os.path.join(MOVIELENS_DIR, USER_DATA_FILE), 
#                     sep='::', 
#                     engine='python', 
#                     encoding='latin-1',
#                     names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],
#                     usecols=['user_id', 'gender', 'age', 'occupation'])

# movies = pd.read_csv(os.path.join(MOVIELENS_DIR, MOVIE_DATA_FILE), 
#                     sep='::', 
#                     engine='python', 
#                     encoding='latin-1',
#                     names=['movie_id', 'title', 'genres'])

# df = ratings.merge(users, on='user_id').merge(movies, on='movie_id')

# df['user_id'] = df['user_id'] - 1
# df['movie_id'] = df['movie_id'] - 1

# nusers = df['user_id'].max() + 1    # 6040
# nmovies = df['movie_id'].max() + 1  # 3952

# # extract years
# df['date'] = pd.to_datetime(df['timestamp'], unit='s')
# df['year_rating'] = pd.DatetimeIndex(df['date']).year
# df['year_movie'] = df['title'].str.extract(r'\((\d+)\)').astype('int64')
# df['genre'] = df['genres'].transform(lambda s: s.split('|')[0])

# # df = df.drop(columns=['timestamp', 'date', 'title', 'genres'])
# df = df.drop(columns=['timestamp', 'date', 'genres'])

# cols = ['gender', 'age', 'year_rating', 'year_movie', 'genre']
# cat2is = []  # category to int
# for col in cols:
#     cats = sorted(df[col].unique().tolist())
#     cat2i = {cat: i for i, cat in enumerate(cats)}
#     cat2is.append(cat2i)
#     df[col] = df[col].transform(lambda cat: cat2i[cat])

# counts = [nusers, nmovies] + [len(cat2i) for cat2i in cat2is[:2]] + [len(df['occupation'].unique())] + [len(cat2i) for cat2i in cat2is[2:]]
# print("Category counts of 'user_id','movie_id','gender','age','occupation','year_rating','year_movie','genre':")
# print(counts)  # [6040, 3952, 2, 7, 21, 4, 81, 18]

# df.to_csv('ml-1m.csv', index=False)

# # Preprocess IMDB
# dfs = []
# dfs.append(pd.read_csv('imdb/title.basics.tsv', sep='\t'))
# dfs.append(pd.read_csv('imdb/title.principals.tsv', sep='\t'))
# dfs.append(pd.read_csv('imdb/title.ratings.tsv', sep='\t'))

# ddf = dfs[0]  # basics
# ddf = ddf.replace(r'\N', np.NaN)
# ddf = ddf.dropna(subset=['startYear', 'runtimeMinutes'])
# ddf[['startYear']] = ddf[['startYear']].astype(int)
# ddf[['runtimeMinutes']] = ddf[['runtimeMinutes']].astype(int)
# ddf = ddf.loc[ (ddf['titleType'] == 'movie') & (ddf['startYear'] >= 1919) & (ddf['startYear'] <= 2000) ]
# for index, row in ddf.iterrows():
#     ddf.loc[index, 'primaryTitle'] = '{} ({})'.format(row['primaryTitle'], row['startYear'])  # very time-consuming
# dfs[0] = ddf

# dfs[0].to_csv('retitled_basics.csv', index=False)

# ddf = dfs[0]  # basics
# dddf = dfs[2]  # ratings
# ddf = ddf.merge(dddf, on='tconst')
# ddf = ddf[['runtimeMinutes','numVotes','primaryTitle']]

# ddf.to_csv('imdb.csv', index=False)

# # Merge two parties
# df = pd.read_csv('ml-1m.csv')
# ddf = pd.read_csv('imdb.csv')

# aligned_df = pd.merge(df, ddf, left_on='title', right_on='primaryTitle')
# aligned_df = aligned_df.drop(columns=['title', 'primaryTitle'])

# aligned_df.to_csv('aligned.csv', index=False)

# union_df = pd.merge(df, ddf, left_on='title', right_on='primaryTitle', how='left')
# union_df = union_df.drop(columns=['title', 'primaryTitle'])

# union_df.to_csv('union.csv', index=False)

ml1m_df = pd.read_csv('ml-1m.csv')
imdb_df = pd.read_csv('imdb.csv')
aligned_df = pd.read_csv('aligned.csv')
union_df = pd.read_csv('union.csv')
counts = [6040, 3952, 2, 7, 21, 4, 81, 18]

aligned_df = aligned_df.sample(frac=1., random_state=seed)
union_df = union_df.sample(frac=1., random_state=seed)
ml1m_df = ml1m_df.sample(frac=1., random_state=seed)

ml1m_full_df = ml1m_df.drop(columns=['title'])
ml1m_aligned_df = aligned_df.drop(columns=['runtimeMinutes', 'numVotes'])

print('dataframes:', ml1m_full_df.shape, ml1m_aligned_df.shape, aligned_df.shape, union_df.shape)  # only need ml1m_full_df and aligned_df

# aligned data normalization
df_train, df_test = train_test_split(aligned_df, test_size=0.2, random_state=seed)
X_train = df_train[['runtimeMinutes', 'numVotes']].to_numpy()
X = aligned_df[['runtimeMinutes', 'numVotes']].to_numpy()
scaler = StandardScaler()
X_train = scaler.fit(X_train)
X = scaler.transform(X)
aligned_df[['runtimeMinutes', 'numVotes']] = X

# union data normalization
df_train, df_test = train_test_split(union_df, test_size=0.2, random_state=seed)
X_train = df_train[['runtimeMinutes', 'numVotes']].to_numpy()
X = union_df[['runtimeMinutes', 'numVotes']].to_numpy()
union_scaler = StandardScaler()
X_train = union_scaler.fit(X_train)
X = union_scaler.transform(X)
union_df[['runtimeMinutes', 'numVotes']] = X
union_df = union_df.fillna(0)


# Single model

class DLRM(nn.Module):
    def __init__(self, top_mlp_units, bot_mlp_units, emb_dim, counts, num_dense):
        super().__init__()
        num_cat = len(counts)
        self.emb_dim = emb_dim
        self.num_cat = num_cat
        self.num_dense = num_dense
        
        embs = [nn.Embedding(cnt, emb_dim) for cnt in counts]
        self.embs = nn.ModuleList(embs)
        
        if self.num_dense > 0:
            bot_mlp = []
            prev = num_dense
            for units in bot_mlp_units:
                bot_mlp.append(nn.Linear(prev, units))
                bot_mlp.append(nn.ReLU())
                prev = units
            bot_mlp.append(nn.Linear(prev, emb_dim))
            self.bot_mlp = nn.Sequential(*bot_mlp)
        
        top_mlp = []
        if self.num_dense > 0:
            prev = emb_dim * (num_cat + 1)
        else:
            prev = emb_dim * num_cat
        for units in top_mlp_units:
            top_mlp.append(nn.Linear(prev, units))
            top_mlp.append(nn.ReLU())
            prev = units
        top_mlp.append(nn.Linear(prev, 1))
        self.top_mlp = nn.Sequential(*top_mlp)
        
        self._initialize_weights()

    def forward(self, inputs):  # inputs: [1d categorical feature, ..., nd dense features]
        embs = []
        
        for i in range(self.num_cat):
            emb = self.embs[i](inputs[i])
            embs.append(emb)
            
        if self.num_dense > 0:
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
                
def train():
    model.train()
    trn_loss = 0.0
    trn_total = 0
    for data in trnloader:
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
    trn_loss /= trn_total
    return trn_loss

def test(dataloader):
    model.eval()
    vld_loss = 0.0
    vld_total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = [x.to(device) for x in data[:-1]]
            labels = data[-1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            cnt = inputs[0].size(0)
            vld_total += cnt
            vld_loss += loss.item() * cnt
    vld_loss /= vld_total
    return vld_loss


# # aligned ; pytorch

# epochs = 10
# batch_size = 128
# lr = 1e-3

# df_trnvld, df_tst = train_test_split(aligned_df, test_size=0.2, random_state=seed)
# df_trn, df_vld = train_test_split(df_trnvld, test_size=0.2, random_state=seed)

# x_trn = [df_trn.user_id, df_trn.movie_id, df_trn.gender, df_trn.age,
#          df_trn.occupation, df_trn.year_rating, df_trn.year_movie, df_trn.genre,
#          df_trn[['runtimeMinutes', 'numVotes']].astype('float32')]
# x_vld = [df_vld.user_id, df_vld.movie_id, df_vld.gender, df_vld.age,
#          df_vld.occupation, df_vld.year_rating, df_vld.year_movie, df_vld.genre,
#          df_vld[['runtimeMinutes', 'numVotes']].astype('float32')]
# x_tst = [df_tst.user_id, df_tst.movie_id, df_tst.gender, df_tst.age,
#          df_tst.occupation, df_tst.year_rating, df_tst.year_movie, df_tst.genre,
#          df_tst[['runtimeMinutes', 'numVotes']].astype('float32')]

# x_trn = [torch.from_numpy(col.to_numpy()) for col in x_trn]
# x_vld = [torch.from_numpy(col.to_numpy()) for col in x_vld]
# x_tst = [torch.from_numpy(col.to_numpy()) for col in x_tst]
# y_trn = torch.from_numpy(df_trn.rating.astype('float32').to_numpy())
# y_vld = torch.from_numpy(df_vld.rating.astype('float32').to_numpy())
# y_tst = torch.from_numpy(df_tst.rating.astype('float32').to_numpy())

# trnset = torch.utils.data.TensorDataset(*x_trn, y_trn)
# vldset = torch.utils.data.TensorDataset(*x_vld, y_vld)
# tstset = torch.utils.data.TensorDataset(*x_tst, y_tst)

# trnloader = torch.utils.data.DataLoader(trnset, batch_size=batch_size, shuffle=True)
# vldloader = torch.utils.data.DataLoader(vldset, batch_size=batch_size, shuffle=False)
# tstloader = torch.utils.data.DataLoader(tstset, batch_size=batch_size, shuffle=False)

# top_mlp_units = [256, 128, 64]
# bot_mlp_units = []
# emb_dim = 16
# num_dense = 2

# model = DLRM(top_mlp_units, bot_mlp_units, emb_dim, counts, num_dense)
# summary(model, next(iter(trnloader))[:-1])

# model = model.to(device)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters())

# print('aligned')
# for epoch in range(epochs):
#     start_t = time()
#     trn_loss = train()
#     vld_loss = test(vldloader)
#     tst_loss = test(tstloader)
#     end_t = time()
#     print('Epoch %d trn_loss: %.4f vld_loss: %.4f tst_loss: %.4f Time: %d s' %
#           (epoch, trn_loss, vld_loss, tst_loss, end_t-start_t))

    
# union ; pytorch

epochs = 10
batch_size = 128
lr = 1e-3

df_trnvld, df_tst = train_test_split(union_df, test_size=0.2, random_state=seed)
df_trn, df_vld = train_test_split(df_trnvld, test_size=0.2, random_state=seed)

x_trn = [df_trn.user_id, df_trn.movie_id, df_trn.gender, df_trn.age,
         df_trn.occupation, df_trn.year_rating, df_trn.year_movie, df_trn.genre,
         df_trn[['runtimeMinutes', 'numVotes']].astype('float32')]
x_vld = [df_vld.user_id, df_vld.movie_id, df_vld.gender, df_vld.age,
         df_vld.occupation, df_vld.year_rating, df_vld.year_movie, df_vld.genre,
         df_vld[['runtimeMinutes', 'numVotes']].astype('float32')]
x_tst = [df_tst.user_id, df_tst.movie_id, df_tst.gender, df_tst.age,
         df_tst.occupation, df_tst.year_rating, df_tst.year_movie, df_tst.genre,
         df_tst[['runtimeMinutes', 'numVotes']].astype('float32')]

x_trn = [torch.from_numpy(col.to_numpy()) for col in x_trn]
x_vld = [torch.from_numpy(col.to_numpy()) for col in x_vld]
x_tst = [torch.from_numpy(col.to_numpy()) for col in x_tst]
y_trn = torch.from_numpy(df_trn.rating.astype('float32').to_numpy())
y_vld = torch.from_numpy(df_vld.rating.astype('float32').to_numpy())
y_tst = torch.from_numpy(df_tst.rating.astype('float32').to_numpy())

trnset = torch.utils.data.TensorDataset(*x_trn, y_trn)
vldset = torch.utils.data.TensorDataset(*x_vld, y_vld)
tstset = torch.utils.data.TensorDataset(*x_tst, y_tst)

trnloader = torch.utils.data.DataLoader(trnset, batch_size=batch_size, shuffle=True)
vldloader = torch.utils.data.DataLoader(vldset, batch_size=batch_size, shuffle=False)
tstloader = torch.utils.data.DataLoader(tstset, batch_size=batch_size, shuffle=False)

top_mlp_units = [256, 128, 64]
bot_mlp_units = []
emb_dim = 16
num_dense = 2

model = DLRM(top_mlp_units, bot_mlp_units, emb_dim, counts, num_dense)
summary(model, next(iter(trnloader))[:-1])

model = model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

### TIME ###

print('union')
for epoch in range(epochs):
    start_t = time()
    trn_loss = train()
    vld_loss = test(vldloader)
    tst_loss = test(tstloader)
    end_t = time()
    print('Epoch %d trn_loss: %.4f vld_loss: %.4f tst_loss: %.4f Time: %d s' %
          (epoch, trn_loss, vld_loss, tst_loss, end_t-start_t))

    
# # ml-1m aligned ; pytorch

# epochs = 10
# batch_size = 128
# lr = 1e-3

# df_trnvld, df_tst = train_test_split(ml1m_aligned_df, test_size=0.2, random_state=seed)
# df_trn, df_vld = train_test_split(df_trnvld, test_size=0.2, random_state=seed)

# x_trn = [df_trn.user_id, df_trn.movie_id, df_trn.gender, df_trn.age,
#          df_trn.occupation, df_trn.year_rating, df_trn.year_movie, df_trn.genre]
# x_vld = [df_vld.user_id, df_vld.movie_id, df_vld.gender, df_vld.age,
#          df_vld.occupation, df_vld.year_rating, df_vld.year_movie, df_vld.genre]
# x_tst = [df_tst.user_id, df_tst.movie_id, df_tst.gender, df_tst.age,
#          df_tst.occupation, df_tst.year_rating, df_tst.year_movie, df_tst.genre]

# x_trn = [torch.from_numpy(col.to_numpy()) for col in x_trn]
# x_vld = [torch.from_numpy(col.to_numpy()) for col in x_vld]
# x_tst = [torch.from_numpy(col.to_numpy()) for col in x_tst]
# y_trn = torch.from_numpy(df_trn.rating.astype('float32').to_numpy())
# y_vld = torch.from_numpy(df_vld.rating.astype('float32').to_numpy())
# y_tst = torch.from_numpy(df_tst.rating.astype('float32').to_numpy())

# trnset = torch.utils.data.TensorDataset(*x_trn, y_trn)
# vldset = torch.utils.data.TensorDataset(*x_vld, y_vld)
# tstset = torch.utils.data.TensorDataset(*x_tst, y_tst)

# trnloader = torch.utils.data.DataLoader(trnset, batch_size=batch_size, shuffle=True)
# vldloader = torch.utils.data.DataLoader(vldset, batch_size=batch_size, shuffle=False)
# tstloader = torch.utils.data.DataLoader(tstset, batch_size=batch_size, shuffle=False)

# top_mlp_units = [256, 128, 64]
# emb_dim = 16
# num_dense = 0

# model = DLRM(top_mlp_units, bot_mlp_units, emb_dim, counts, num_dense)
# summary(model, next(iter(trnloader))[:-1])

# model = model.to(device)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters())

# print('ml-1m aligned')
# for epoch in range(epochs):
#     start_t = time()
#     trn_loss = train()
#     vld_loss = test(vldloader)
#     tst_loss = test(tstloader)
#     end_t = time()
#     print('Epoch %d trn_loss: %.4f vld_loss: %.4f tst_loss: %.4f Time: %d s' %
#           (epoch, trn_loss, vld_loss, tst_loss, end_t-start_t))

    
# ml-1m full ; pytorch

epochs = 10
batch_size = 128
lr = 1e-3

df_trnvld, df_tst = train_test_split(ml1m_full_df, test_size=0.2, random_state=seed)
df_trn, df_vld = train_test_split(df_trnvld, test_size=0.2, random_state=seed)

x_trn = [df_trn.user_id, df_trn.movie_id, df_trn.gender, df_trn.age,
         df_trn.occupation, df_trn.year_rating, df_trn.year_movie, df_trn.genre]
x_vld = [df_vld.user_id, df_vld.movie_id, df_vld.gender, df_vld.age,
         df_vld.occupation, df_vld.year_rating, df_vld.year_movie, df_vld.genre]
x_tst = [df_tst.user_id, df_tst.movie_id, df_tst.gender, df_tst.age,
         df_tst.occupation, df_tst.year_rating, df_tst.year_movie, df_tst.genre]

x_trn = [torch.from_numpy(col.to_numpy()) for col in x_trn]
x_vld = [torch.from_numpy(col.to_numpy()) for col in x_vld]
x_tst = [torch.from_numpy(col.to_numpy()) for col in x_tst]
y_trn = torch.from_numpy(df_trn.rating.astype('float32').to_numpy())
y_vld = torch.from_numpy(df_vld.rating.astype('float32').to_numpy())
y_tst = torch.from_numpy(df_tst.rating.astype('float32').to_numpy())

trnset = torch.utils.data.TensorDataset(*x_trn, y_trn)
vldset = torch.utils.data.TensorDataset(*x_vld, y_vld)
tstset = torch.utils.data.TensorDataset(*x_tst, y_tst)

trnloader = torch.utils.data.DataLoader(trnset, batch_size=batch_size, shuffle=True)
vldloader = torch.utils.data.DataLoader(vldset, batch_size=batch_size, shuffle=False)
tstloader = torch.utils.data.DataLoader(tstset, batch_size=batch_size, shuffle=False)

top_mlp_units = [256, 128, 64]
emb_dim = 16
num_dense = 0

model = DLRM(top_mlp_units, bot_mlp_units, emb_dim, counts, num_dense)
summary(model, next(iter(trnloader))[:-1])

model = model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

### TIME ###

print('ml-1m full')
for epoch in range(epochs):
    start_t = time()
    trn_loss = train()
    vld_loss = test(vldloader)
    tst_loss = test(tstloader)
    end_t = time()
    print('Epoch %d trn_loss: %.4f vld_loss: %.4f tst_loss: %.4f Time: %d s' %
          (epoch, trn_loss, vld_loss, tst_loss, end_t-start_t))


### SplitNN ###

class ClientNet(nn.Module):
    def __init__(self, bot_mlp_units, emb_dim, counts, num_dense):
        super().__init__()
        num_cat = len(counts)
        self.emb_dim = emb_dim
        self.num_cat = num_cat
        self.num_dense = num_dense
        
        if num_cat > 0:
            embs = [nn.Embedding(cnt, emb_dim) for cnt in counts]
            self.embs = nn.ModuleList(embs)
        
        if self.num_dense > 0:
            bot_mlp = []
            prev = num_dense
            for units in bot_mlp_units:
                bot_mlp.append(nn.Linear(prev, units))
                bot_mlp.append(nn.ReLU())
                prev = units
            bot_mlp.append(nn.Linear(prev, emb_dim))
            self.bot_mlp = nn.Sequential(*bot_mlp)
        
        self._initialize_weights()

    def forward(self, inputs):  # inputs: [1d categorical feature, ..., nd dense features]
        embs = []
        
        for i in range(self.num_cat):
            emb = self.embs[i](inputs[i])
            embs.append(emb)
        
        if self.num_dense > 0:
            dense_emb = self.bot_mlp(inputs[-1])
            embs.append(dense_emb)
        
        out = torch.cat(embs, dim=1)
        
        return out
    
    def _initialize_weights(self):  # same as keras
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.05, 0.05)


class ServerNet(nn.Module):
    def __init__(self, top_mlp_units, input_units):
        super().__init__()
        top_mlp = []
        prev = input_units
        for units in top_mlp_units:
            top_mlp.append(nn.Linear(prev, units))
            top_mlp.append(nn.ReLU())
            prev = units
        top_mlp.append(nn.Linear(prev, 1))
        self.top_mlp = nn.Sequential(*top_mlp)
        self._initialize_weights()

    def forward(self, inputs):
        x = torch.cat(tuple(inputs), dim=1)
        x = self.top_mlp(x)
        x = torch.flatten(x)
        return x
    
    def _initialize_weights(self):  # same as keras
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.05, 0.05)

                
# # aligned ; splitnn ; pytorch

# epochs = 10
# batch_size = 128
# lr = 1e-3

# df_trnvld, df_tst = train_test_split(aligned_df, test_size=0.2, random_state=seed)
# df_trn, df_vld = train_test_split(df_trnvld, test_size=0.2, random_state=seed)

# x_trn = [df_trn.user_id, df_trn.movie_id, df_trn.gender, df_trn.age,
#          df_trn.occupation, df_trn.year_rating, df_trn.year_movie, df_trn.genre,
#          df_trn[['runtimeMinutes', 'numVotes']].astype('float32')]
# x_vld = [df_vld.user_id, df_vld.movie_id, df_vld.gender, df_vld.age,
#          df_vld.occupation, df_vld.year_rating, df_vld.year_movie, df_vld.genre,
#          df_vld[['runtimeMinutes', 'numVotes']].astype('float32')]
# x_tst = [df_tst.user_id, df_tst.movie_id, df_tst.gender, df_tst.age,
#          df_tst.occupation, df_tst.year_rating, df_tst.year_movie, df_tst.genre,
#          df_tst[['runtimeMinutes', 'numVotes']].astype('float32')]

# x_trn = [torch.from_numpy(col.to_numpy()) for col in x_trn]
# x_vld = [torch.from_numpy(col.to_numpy()) for col in x_vld]
# x_tst = [torch.from_numpy(col.to_numpy()) for col in x_tst]
# y_trn = torch.from_numpy(df_trn.rating.astype('float32').to_numpy())
# y_vld = torch.from_numpy(df_vld.rating.astype('float32').to_numpy())
# y_tst = torch.from_numpy(df_tst.rating.astype('float32').to_numpy())

# trnset = torch.utils.data.TensorDataset(*x_trn, y_trn)
# vldset = torch.utils.data.TensorDataset(*x_vld, y_vld)
# tstset = torch.utils.data.TensorDataset(*x_tst, y_tst)

# trnloader = torch.utils.data.DataLoader(trnset, batch_size=batch_size, shuffle=True)
# vldloader = torch.utils.data.DataLoader(vldset, batch_size=batch_size, shuffle=False)
# tstloader = torch.utils.data.DataLoader(tstset, batch_size=batch_size, shuffle=False)

# top_mlp_units = [256, 128, 64]
# bot_mlp_units = []
# emb_dim = 16
# num_dense = 2
# num_cat = len(counts)

# def train():
#     for model in models:
#         model.train()
#     trn_loss = 0.0
#     trn_total = 0
#     for data in trnloader:
#         ml1m_input = [x.to(device) for x in data[:num_cat]]
#         imdb_input = [x.to(device) for x in data[num_cat:-1]]
#         labels = data[-1].to(device)
        
#         with torch.set_grad_enabled(True):
#             for optim in optims:
#                 optim.zero_grad()
#             ml1m_client_out = ml1m_client_net(ml1m_input)
#             imdb_client_out = imdb_client_net(imdb_input)
#             prediction_clients = [ml1m_client_out, imdb_client_out]

#             # send tensors to server, server continue prediction
#             # TODO: the copy causes buffer overflow
#             prediction_clients_copy = [torch.zeros_like(prediction_client) for prediction_client in prediction_clients]
#             for prediction_copy, prediction_client in zip(prediction_clients_copy, prediction_clients):
#                 prediction_copy.data.copy_(prediction_client)
#                 prediction_copy.requires_grad_(True)
#             prediction = ml1m_server_net(prediction_clients_copy)
#             loss = criterion(prediction, labels)
#             loss.backward()
#             ml1m_server_optim.step()

#             # server send back the split layer (prediction_clients_copy) to client, client continue back prop
#             for prediction_client, prediction_copy in zip(prediction_clients, prediction_clients_copy):
#                 prediction_client.backward(prediction_copy.grad)
#             ml1m_client_optim.step()
#             imdb_client_optim.step()
        
#             cnt = ml1m_input[0].size(0)
#             trn_total += cnt
#             trn_loss += loss.item() * cnt
            
#     trn_loss /= trn_total
#     return trn_loss

# def test(dataloader):
#     for model in models:
#         model.eval()
#     vld_loss = 0.0
#     vld_total = 0
#     with torch.no_grad():
#         for data in dataloader:
#             ml1m_input = [x.to(device) for x in data[:num_cat]]
#             imdb_input = [x.to(device) for x in data[num_cat:-1]]
#             labels = data[-1].to(device)
            
#             with torch.set_grad_enabled(False):
#                 ml1m_client_out = ml1m_client_net(ml1m_input)
#                 imdb_client_out = imdb_client_net(imdb_input)
#                 prediction_clients = [ml1m_client_out, imdb_client_out]
#                 prediction = ml1m_server_net(prediction_clients)
#                 loss = criterion(prediction, labels)
                
#                 cnt = ml1m_input[0].size(0)
#                 vld_total += cnt
#                 vld_loss += loss.item() * cnt
                
#     vld_loss /= vld_total
#     return vld_loss

# top_in = emb_dim * (len(counts) + 1)
# ml1m_client_net = ClientNet(bot_mlp_units=[], emb_dim=emb_dim, counts=counts, num_dense=0)
# imdb_client_net = ClientNet(bot_mlp_units=bot_mlp_units, emb_dim=emb_dim, counts=[], num_dense=num_dense)
# ml1m_server_net = ServerNet(top_mlp_units, top_in)

# summary(ml1m_client_net, next(iter(trnloader))[:len(counts)])
# summary(imdb_client_net, next(iter(trnloader))[len(counts):-1])
# summary(ml1m_server_net, [torch.zeros(batch_size, emb_dim * len(counts)), torch.zeros(batch_size, emb_dim)])

# criterion = torch.nn.MSELoss().to(device)
# ml1m_client_net = ml1m_client_net.to(device)
# imdb_client_net = imdb_client_net.to(device)
# ml1m_server_net = ml1m_server_net.to(device)
# ml1m_client_optim = torch.optim.Adam(ml1m_client_net.parameters())
# imdb_client_optim = torch.optim.Adam(imdb_client_net.parameters())
# ml1m_server_optim = torch.optim.Adam(ml1m_server_net.parameters())
# models = [ml1m_client_net, imdb_client_net, ml1m_server_net]
# optims = [ml1m_client_optim, imdb_client_optim, ml1m_server_optim]

# print('aligned ; splitnn')
# for epoch in range(epochs):
#     start_t = time()
#     trn_loss = train()
#     vld_loss = test(vldloader)
#     tst_loss = test(tstloader)
#     end_t = time()
#     print('Epoch %d trn_loss: %.4f vld_loss: %.4f tst_loss: %.4f Time: %d s' %
#           (epoch, trn_loss, vld_loss, tst_loss, end_t-start_t))

    
# union ; splitnn ; pytorch

epochs = 10
batch_size = 128
lr = 1e-3

df_trnvld, df_tst = train_test_split(union_df, test_size=0.2, random_state=seed)
df_trn, df_vld = train_test_split(df_trnvld, test_size=0.2, random_state=seed)

x_trn = [df_trn.user_id, df_trn.movie_id, df_trn.gender, df_trn.age,
         df_trn.occupation, df_trn.year_rating, df_trn.year_movie, df_trn.genre,
         df_trn[['runtimeMinutes', 'numVotes']].astype('float32')]
x_vld = [df_vld.user_id, df_vld.movie_id, df_vld.gender, df_vld.age,
         df_vld.occupation, df_vld.year_rating, df_vld.year_movie, df_vld.genre,
         df_vld[['runtimeMinutes', 'numVotes']].astype('float32')]
x_tst = [df_tst.user_id, df_tst.movie_id, df_tst.gender, df_tst.age,
         df_tst.occupation, df_tst.year_rating, df_tst.year_movie, df_tst.genre,
         df_tst[['runtimeMinutes', 'numVotes']].astype('float32')]

x_trn = [torch.from_numpy(col.to_numpy()) for col in x_trn]
x_vld = [torch.from_numpy(col.to_numpy()) for col in x_vld]
x_tst = [torch.from_numpy(col.to_numpy()) for col in x_tst]
y_trn = torch.from_numpy(df_trn.rating.astype('float32').to_numpy())
y_vld = torch.from_numpy(df_vld.rating.astype('float32').to_numpy())
y_tst = torch.from_numpy(df_tst.rating.astype('float32').to_numpy())

trnset = torch.utils.data.TensorDataset(*x_trn, y_trn)
vldset = torch.utils.data.TensorDataset(*x_vld, y_vld)
tstset = torch.utils.data.TensorDataset(*x_tst, y_tst)

trnloader = torch.utils.data.DataLoader(trnset, batch_size=batch_size, shuffle=True)
vldloader = torch.utils.data.DataLoader(vldset, batch_size=batch_size, shuffle=False)
tstloader = torch.utils.data.DataLoader(tstset, batch_size=batch_size, shuffle=False)

top_mlp_units = [256, 128, 64]
bot_mlp_units = []
emb_dim = 16
num_dense = 2
num_cat = len(counts)

### TIME ###

def train():
    for model in models:
        model.train()
    trn_loss = 0.0
    trn_total = 0
    for data in trnloader:
        ml1m_input = [x.to(device) for x in data[:num_cat]]
        imdb_input = [x.to(device) for x in data[num_cat:-1]]
        labels = data[-1].to(device)
        
        with torch.set_grad_enabled(True):
            for optim in optims:
                optim.zero_grad()
            ml1m_client_out = ml1m_client_net(ml1m_input)
            imdb_client_out = imdb_client_net(imdb_input)
            prediction_clients = [ml1m_client_out, imdb_client_out]

            # send tensors to server, server continue prediction
            # TODO: the copy causes buffer overflow
            prediction_clients_copy = [torch.zeros_like(prediction_client) for prediction_client in prediction_clients]
            for prediction_copy, prediction_client in zip(prediction_clients_copy, prediction_clients):
                prediction_copy.data.copy_(prediction_client)
                prediction_copy.requires_grad_(True)
            prediction = ml1m_server_net(prediction_clients_copy)
            loss = criterion(prediction, labels)
            loss.backward()
            ml1m_server_optim.step()

            # server send back the split layer (prediction_clients_copy) to client, client continue back prop
            for prediction_client, prediction_copy in zip(prediction_clients, prediction_clients_copy):
                prediction_client.backward(prediction_copy.grad)
            ml1m_client_optim.step()
            imdb_client_optim.step()
        
            cnt = ml1m_input[0].size(0)
            trn_total += cnt
            trn_loss += loss.item() * cnt
            
    trn_loss /= trn_total
    return trn_loss

def test(dataloader):
    for model in models:
        model.eval()
    vld_loss = 0.0
    vld_total = 0
    with torch.no_grad():
        for data in dataloader:
            ml1m_input = [x.to(device) for x in data[:num_cat]]
            imdb_input = [x.to(device) for x in data[num_cat:-1]]
            labels = data[-1].to(device)
            
            with torch.set_grad_enabled(False):
                ml1m_client_out = ml1m_client_net(ml1m_input)
                imdb_client_out = imdb_client_net(imdb_input)
                prediction_clients = [ml1m_client_out, imdb_client_out]
                prediction = ml1m_server_net(prediction_clients)
                loss = criterion(prediction, labels)
                
                cnt = ml1m_input[0].size(0)
                vld_total += cnt
                vld_loss += loss.item() * cnt
                
    vld_loss /= vld_total
    return vld_loss

top_in = emb_dim * (len(counts) + 1)
ml1m_client_net = ClientNet(bot_mlp_units=[], emb_dim=emb_dim, counts=counts, num_dense=0)
imdb_client_net = ClientNet(bot_mlp_units=bot_mlp_units, emb_dim=emb_dim, counts=[], num_dense=num_dense)
ml1m_server_net = ServerNet(top_mlp_units, top_in)

summary(ml1m_client_net, next(iter(trnloader))[:len(counts)])
summary(imdb_client_net, next(iter(trnloader))[len(counts):-1])
summary(ml1m_server_net, [torch.zeros(batch_size, emb_dim * len(counts)), torch.zeros(batch_size, emb_dim)])

criterion = torch.nn.MSELoss().to(device)
ml1m_client_net = ml1m_client_net.to(device)
imdb_client_net = imdb_client_net.to(device)
ml1m_server_net = ml1m_server_net.to(device)
ml1m_client_optim = torch.optim.Adam(ml1m_client_net.parameters())
imdb_client_optim = torch.optim.Adam(imdb_client_net.parameters())
ml1m_server_optim = torch.optim.Adam(ml1m_server_net.parameters())
models = [ml1m_client_net, imdb_client_net, ml1m_server_net]
optims = [ml1m_client_optim, imdb_client_optim, ml1m_server_optim]

### TIME ###

print('union ; splitnn')
for epoch in range(epochs):
    start_t = time()
    trn_loss = train()
    vld_loss = test(vldloader)
    tst_loss = test(tstloader)
    end_t = time()
    print('Epoch %d trn_loss: %.4f vld_loss: %.4f tst_loss: %.4f Time: %d s' %
          (epoch, trn_loss, vld_loss, tst_loss, end_t-start_t))
