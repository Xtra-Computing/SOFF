import csv
import re

cnt = 0
cleaned = {}
with open('ign.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    head = next(csv_reader)
    for row in csv_reader:
        title = row[2]
        title = title.lower().replace(' ', '')
        title = ''.join(filter(lambda x: x.isalnum(), title))

        if title not in cleaned:            
            new_row = [row[2], row[1], row[5], row[6], row[7]]
            cleaned[title] = new_row

        cnt += 1

print('cleaned number', len(cleaned))
print('total number', cnt)

with open('ign_game_clean.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow([head[2], head[1], head[5], head[6], head[7]])
    for k in cleaned:
        writer.writerow(cleaned[k])