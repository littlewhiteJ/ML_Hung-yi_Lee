# validation data preprocess
import json

with open('test.csv', 'r', encoding="utf8", errors='ignore') as f:
    lines = f.readlines()

d = {}
for i in range(0, len(lines)):
    line = lines[i]
    line = line.split(',')

    dataID = line[0]
    if dataID not in d:
        d[dataID] = {}

    tag = line[1]
    if tag not in d[dataID]:
        d[dataID][tag] = []    
    for j in range(2, 11):
        if line[j].strip() != 'NR':
            d[dataID][tag].append(float(line[j]))
        else:
            d[dataID][tag].append(0.0)


'''
for tag in d:
    print(d[tag][:50])
'''
with open('validData.json','w') as f:
    json.dump(d, f)
