from regress import *
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from utils import *

with open('config.json', 'r') as f:
    config = json.load(f)

name = config['name']
logName = 'log/' + name + '.txt'
modelName = 'model/' + name + '.json'
trainFile = config['trainFile']
validationFile = config['validationFile']
norDFile = config['norDFile']

epoch = int(config['epoch'])
ifRandom = bool(config['ifRandom'])
trainNum = int(config['trainNum'])
allNum = int(config['allNum'])
t = int(config['t'])
maxPower = int(config['maxPower'])
lr = float(config['lr'])
alpha = float(config['alpha'])
checkEpoch = int(config['checkEpoch'])
checkNum = int(config['checkNum'])


with open(validationFile, 'r') as f:
    d = json.load(f)

with open(modelName, 'r') as f:
    dModel = json.load(f)

with open(norDFile, 'r') as f:
    norD = json.load(f)

d = normalizeV(d, norD)

f4 = F4(lr, t, maxPower, alpha)
f4.setParameter(np.array(dModel['w']), np.array(dModel['b']))

with open('data/ans.txt', 'w') as f:
    f.write('id,value\n')
    for dataID in d:
        dd = d[dataID]
        x = []
        for tag in dd:
            x.append(dd[tag])
        x = np.array(x)
        y = round(denormalize(f4.predict(x), norD), 2)
        f.write(dataID + ',' + str(y) + '\n')
        


'''
plt.plot(showbar[2:])
plt.show()
'''
# print(f.parameter())
