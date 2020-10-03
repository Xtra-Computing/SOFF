import random
import numpy as np
import pandas as pd
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
set_seed(0)

interact_df = pd.read_csv('align_interact.csv')

align_info = pd.read_csv('align_info.csv')
app_id = interact_df['appid'].unique().tolist()

grouped = interact_df.groupby('steamid')
negative = []
partial_cnt = 0
ratio = 3
cnt = 0
print('start')
for index, value in grouped:
    exist_id = set(value['appid'])
    length = len(exist_id)

    if length*(ratio+1) >= len(app_id):
        for a in app_id:
            if a not in exist_id:
                negative.append([index, a])
                partial_cnt += 1
    else:
        while len(exist_id) != length*(ratio+1):
            new_id = app_id[random.randint(0, len(app_id)-1)]
            if new_id not in exist_id:
                exist_id.add(new_id)
                negative.append([index, new_id])   
    cnt += 1
    if cnt % 1000 == 0:
        print('current step', cnt)

print('group cnt', cnt)
print('partial cnt', partial_cnt)
print('negative sample cnt', len(negative))

negative_df = pd.DataFrame(negative, columns=['steamid', 'appid'])
negative_df['label'] = list(np.zeros(negative_df.shape[0], dtype=np.int))

interact_df['label'] = list(np.ones(interact_df.shape[0], dtype=np.int))

all_interact_df = interact_df.append(negative_df)
print('all interaction shape', all_interact_df.shape)


steam_game = pd.read_csv('steam_game_clean.csv')
ign_game = pd.read_csv('ign_game_clean.csv')
ign_game['ign_index'] = range(ign_game.shape[0])

ign_align = pd.merge(align_info, ign_game, on='ign_index', sort=False)
ign_align = ign_align.drop(columns=['title', 'steam_index', 'ign_index'])

steam_data_df = pd.merge(steam_game, all_interact_df, on='appid', sort=False)
two_party_data_df = pd.merge(ign_align, steam_data_df, on='appid', sort=False)
print('aligned all data shape', two_party_data_df.shape)

two_party_data_df.to_csv('align_all_data.csv', index=False)

