import numpy as np

class logAndShow:
    def __init__(self, fileName):
        self.fileName = fileName
        self.list = []
    
    def print(self, s):
        print(s)
        self.list.append(s)
    
    def close(self):
        with open(self.fileName, 'w') as f:
            for l in self.list:
                f.write(l)
                f.write('\n')

class dataLoader:
    def __init__(self, d, t):
        self.d = d
        self.t = t
    
    def randomLoad(self, dataRange):
        index = np.random.randint(dataRange[0], dataRange[1])
        x = []
        for tag in self.d:
            x.append(self.d[tag][index:index+self.t])
        y = self.d['PM2.5'][index+self.t]
        x = np.array(x)
        return x, y

    def orderedLoad(self, index):
        x = []
        for tag in self.d:
            x.append(self.d[tag][index:index+self.t])
        y = self.d['PM2.5'][index+self.t]
        x = np.array(x)
        return x, y

def normalize(d):
    nd = {}
    normalizeD = {}
    for tag in d:
        l = d[tag]
        nl = []
        lmin = min(l)
        lrange = max(l) - min(l)
        for item in l:
            nl.append((item - lmin) / lrange)
        nd[tag] = nl
        normalizeD[tag] = [lmin, lrange]
    return nd, normalizeD

def denormalize(y, norD):
    lmin, lrange = norD['PM2.5']
    y = y * lrange + lmin
    return y

def errorDenormalize(e, norD):
    lrange = norD['PM2.5'][1]
    return e * lrange

def normalizeV(d, norD):
    nd = {}
    for dataID in d:
        dd = d[dataID]
        ndd = {}
        for tag in dd:
            lmin, lrange = norD[tag]    
            l = dd[tag]
            nl = []
            for item in l:
                nl.append((item - lmin) / lrange)
            ndd[tag] = nl
        nd[dataID] = ndd
    return nd


def RMSE(deltaEs):
    n = len(deltaEs)
    return np.sqrt( np.sum( pow(deltaEs, 2) ) / n )