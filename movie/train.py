# Adapted from https://github.com/khanhnamle1994/movielens
import pandas as pd
import numpy as np
import os
import random as python_random

seed = 0

np.random.seed(seed)
python_random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # uncomment to get reproducible results

import tensorflow as tf
from keras.layers import Input, Embedding, Concatenate, Flatten, Dense, Dot, Add, Multiply, Subtract, Average, Reshape
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session

tf.set_random_seed(seed)
config =  tf.ConfigProto()
config.gpu_options.allow_growth = True
sess =  tf.Session(config=config)
set_session(sess)

MOVIELENS_DIR = 'ml-1m'
USER_DATA_FILE = 'users.dat'
MOVIE_DATA_FILE = 'movies.dat'
RATING_DATA_FILE = 'ratings.dat'

AGES = { 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
OCCUPATIONS = { 0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer" }

ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['user_id', 'movie_id', 'rating', 'timestamp'])

users = pd.read_csv(os.path.join(MOVIELENS_DIR, USER_DATA_FILE), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],
                    usecols=['user_id', 'gender', 'age', 'occupation'])

movies = pd.read_csv(os.path.join(MOVIELENS_DIR, MOVIE_DATA_FILE), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['movie_id', 'title', 'genres'])

df = ratings.merge(users, on='user_id').merge(movies, on='movie_id')

df['user_id'] = df['user_id'] - 1
df['movie_id'] = df['movie_id'] - 1

nusers = df['user_id'].max() + 1    # 6040
nmovies = df['movie_id'].max() + 1  # 3952

# extract years
df['date'] = pd.to_datetime(df['timestamp'], unit='s')
df['year_rating'] = pd.DatetimeIndex(df['date']).year
df['year_movie'] = df['title'].str.extract(r'\((\d+)\)').astype('int64')
df['genre'] = df['genres'].transform(lambda s: s.split('|')[0])

df = df.drop(columns=['timestamp', 'date', 'title', 'genres'])

cols = ['gender', 'age', 'year_rating', 'year_movie', 'genre']
cat2is = []  # category to int
for col in cols:
    cats = sorted(df[col].unique().tolist())
    cat2i = {cat: i for i, cat in enumerate(cats)}
    cat2is.append(cat2i)
    df[col] = df[col].transform(lambda cat: cat2i[cat])

df = df.sample(frac=1., random_state=seed)  # shuffle
nsamples = df.shape[0]
split = int(nsamples * 0.9)
df_train = df.iloc[:split, :]
df_test = df.iloc[split:, :]

def NCF(hidden_units, emb_dim, counts):

    # add embedding for each categorical feature
    inputs = []
    embeddings = []
    for i in range(len(counts)):
        inputs.append(Input(shape=(1,), name=str(i+1)+'_sparse'))
    for j in range(len(counts)):
        embeddings.append(Embedding(counts[j], emb_dim, input_length=1, name=str(j+1)+'_embedding')(inputs[j]))

    # concat the embeddings
    merged = Concatenate()(embeddings)
    out = Flatten()(merged)

    # MLP
    for n_hidden in hidden_units:
        out = Dense(n_hidden, activation='relu')(out)
    out = Dense(1, activation='linear', name='prediction')(out)
    
    model=Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model


print('Setting: Rating only')

counts = [nusers, nmovies]
print("Category counts of 'user_id','movie_id':")
print(counts)  # [6040, 3952]

hidden_units = [32, 16]
emb_dim = 16
ncf = NCF(hidden_units, emb_dim, counts)

x_trn = [df_train.user_id, df_train.movie_id]
x_tst = [df_test.user_id, df_test.movie_id]
y_trn = df_train.rating
y_tst = df_test.rating

es = EarlyStopping(monitor='val_loss', patience=2, verbose=0, restore_best_weights=True)

ncf.fit(x=x_trn, y=y_trn, batch_size=128, epochs=100, verbose=2, validation_split=0.1, callbacks=[es])
loss = ncf.evaluate(x=x_tst, y=y_tst, batch_size=128, verbose=0)
print('Test set MSE:', loss)


print('\nSetting: Rating + Auxiliary')

counts = [nusers, nmovies] + [len(cat2i) for cat2i in cat2is[:2]] + [len(df['occupation'].unique())] + [len(cat2i) for cat2i in cat2is[2:]]
print("Category counts of 'user_id','movie_id','gender','age','occupation','year_rating','year_movie','genre':")
print(counts)  # [6040, 3952, 2, 7, 21, 4, 81, 18]

hidden_units = (256, 128, 64)
emb_dim = 16
ncf = NCF(hidden_units, emb_dim, counts)

x_trn = [df_train.user_id, df_train.movie_id, df_train.gender, df_train.age,
         df_train.occupation, df_train.year_rating, df_train.year_movie, df_train.genre]
x_tst = [df_test.user_id, df_test.movie_id, df_test.gender, df_test.age,
         df_test.occupation, df_test.year_rating, df_test.year_movie, df_test.genre]
y_trn = df_train.rating
y_tst = df_test.rating

es = EarlyStopping(monitor='val_loss', patience=2, verbose=0, restore_best_weights=True)

ncf.fit(x=x_trn, y=y_trn, batch_size=128, epochs=100, verbose=2, validation_split=0.1, callbacks=[es])
loss = ncf.evaluate(x=x_tst, y=y_tst, batch_size=128, verbose=0)
print('Test set MSE:', loss)
