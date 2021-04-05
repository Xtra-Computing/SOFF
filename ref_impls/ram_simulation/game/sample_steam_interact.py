import pandas as pd
interact_df = pd.read_csv('steam_interact.csv', header=None)
interact_df.columns=['steamid', 'appid']
print('interaction shape', interact_df.shape)

steam = pd.read_csv('steam_game_clean.csv')
app_id_df = steam[['appid']]

merged = pd.merge(app_id_df, interact_df, on='appid', sort=False)
print('merged interaction shape', merged.shape)


cnts = merged['steamid'].value_counts()
cnts_df = pd.DataFrame({'steamid': cnts.index, 'cnts': cnts.values})
filters = cnts_df[cnts_df['cnts'] >= 350]
filters_id_df = filters[['steamid']]

sample = pd.merge(filters_id_df, merged, on='steamid', sort=False)
print('subsample shape', sample.shape)

sample.to_csv('steam_interact_sample.csv', index=False)