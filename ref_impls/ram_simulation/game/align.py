import csv
import numpy as np
import pandas as pd

ign_title = set()
ign_map = {}
cnt = 0
with open('ign_game_clean.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for row in csv_reader:
        game = row[0].lower().replace(' ', '')
        game = ''.join(filter(str.isalnum, game))
        ign_title.add(game)
        ign_map[game] = cnt
        cnt += 1

print('ign title num', len(ign_title))

steam_title = set()
steam_map = {}
cnt = 0
with open('steam_game_clean.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for row in csv_reader:
        game = row[0].lower().replace(' ', '')
        game = ''.join(filter(str.isalnum, game))
        steam_title.add(game)
        steam_map[game] = [row[1], cnt]
        cnt += 1

print('steam title num', len(steam_title))


align_title = ign_title.intersection(steam_title)
align_title = list(align_title)
print('align title num', len(align_title))

align_info = []
for a in align_title:
    align_info.append([int(steam_map[a][0]), steam_map[a][1], ign_map[a]])
align_info_df = pd.DataFrame(align_info, columns=['appid','steam_index','ign_index'])
align_info_df.to_csv('align_info.csv', index=False)
align_id_df = align_info_df[['appid']]

interact_df = pd.read_csv('steam_interact_sample.csv')

result = pd.merge(align_id_df, interact_df, on='appid', sort=False)
print('align interaction shape', result.shape)

result.to_csv('align_interact.csv', index=False)


