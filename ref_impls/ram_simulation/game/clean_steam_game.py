import csv

cnt = 0
cleaned = {}
with open('steam-utf8.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        if len(row) > 1:
            str = ''
            for i in row:
                str += i
            row = [str]
  
        row = row[0].split(';"')
        row = [v.replace('\"','') for v in row]
        
        title = row[1]
        if '??' not in title:
            title = title.lower().replace(' ', '')
            title = ''.join(filter(lambda x: x.isalnum(), title))
            if title not in cleaned:   
                new_row = [row[1], row[0], row[2], row[3], row[4][:4], row[6], row[7]]
                cleaned[title] = new_row

        cnt += 1

print('cleaned number', len(cleaned))
print('total number', cnt)

with open('steam_game_clean.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['title', 'appid', 'type', 'price', 'release_year', 'required_age', 'is_multiplayer'])
    for k in cleaned:
        writer.writerow(cleaned[k])