# preprocess for train data
import json

with open('data/train.csv', 'r', encoding="utf8", errors='ignore') as f:
    lines = f.readlines()

d = {}
for i in range(1, len(lines)):
    line = lines[i]
    line = line.split(',')
    tag = line[2]
    if tag not in d:
        d[tag] = []    
    for j in range(3, 27):
        if line[j].strip() != 'NR':
            d[tag].append(float(line[j]))
        else:
            d[tag].append(0.0)


'''
for tag in d:
    print(d[tag][:50])

for tag in d:
    print(tag)
'''

with open('data/rawdata.json','w') as f:
    json.dump(d, f)
