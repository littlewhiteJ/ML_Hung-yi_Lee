from regress import *
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from utils import *

with open('config.json', 'r') as f:
    config = json.load(f)

name = config['name']
logName = 'log/train_' + name + '.txt'
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

log = logAndShow(logName)
log.print(json.dumps(config))

with open(trainFile, 'r') as f:
    d = json.load(f)

d, norD = normalize(d)
log.print(json.dumps(norD))
with open(norDFile, 'w') as f:
    json.dump(norD, f)

f4 = F4(lr, t, maxPower, alpha)
dataloader = dataLoader(d, t)

# showbar = []
if ifRandom:
    for i in range(epoch * trainNum):
        x, y = dataloader.randomLoad((0, trainNum))    
        f4.step(x, y)
        # print(f4.parameter())
        
        if i % checkEpoch == 0:
            es = []
            for j in range(checkNum):
                x, y = dataloader.randomLoad((trainNum, allNum))
                es.append( errorDenormalize( f4.error(x, y), norD ) )
            es = np.array(es)
            rmse = RMSE(es)
            log.print(str(rmse))

else:
    for i in range(epoch):
        for index in range(trainNum):           
            x, y = dataloader.orderedLoad(index)    
            f4.step(x, y)
            # print(f4.parameter())
            
            if i % checkEpoch == 0:
                es = []
                for j in range(checkNum):
                    x, y = dataloader.randomLoad((trainNum, allNum))
                    es.append( errorDenormalize( f4.error(x, y), norD ))
                es = np.array(es)
                rmse = RMSE(es)
                log.print(str(rmse))

# save model parameter
w, b = f4.parameter()
w = w.tolist()
dModel = {}
dModel['w'] = w
dModel['b'] = b

with open(modelName, 'w') as f:
    json.dump(dModel, f)


log.close()
'''
plt.plot(showbar[2:])
plt.show()
'''
# print(f.parameter())
