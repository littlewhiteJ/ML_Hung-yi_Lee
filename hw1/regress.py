import os
import numpy as np
import json
'''
# F0 predict y(PM2.5 now) by x(last PM2.5 data).
# x is one number, here we use the last PM2.5 data.
'''
class F0:
    def __init__(self, lr):
        self.w = 1
        self.b = 1
        self.lr = lr

    def deriv_w(self, x, y): 
        dw = ( y - (x * self.w + self.b) ) * (-2) * x 
        return dw
    
    def deriv_b(self, x, y):
        db = ( y - (x * self.w + self.b) ) * (-2)
        return db

    def step(self, x, y):
        derivW = self.deriv_w(x, y)
        derivB = self.deriv_b(x, y)

        self.w = self.w - self.lr * derivW
        self.b = self.b - self.lr * derivB

        return self.w, self.b
    
    def error(self, x, y):
        e = np.abs(y - (x * self.w + self.b))
        return e

'''
# F1 predict y(PM2.5 now) by x(last 9 PM2.5 data).
# x is a list of 9 number, here we use the last 9 PM2.5 data. 
'''
class F1:
    def __init__(self, lr):
        self.w = np.ones(9)
        self.b = 1
        self.lr = lr

    def deriv_w(self, x, y, i): 
        wx = np.sum(x * self.w)
        dw = ( y - (wx + self.b) ) * (-2) * x[i] 
        return dw
    
    def deriv_b(self, x, y):
        wx = np.sum(x * self.w)
        db = ( y - (wx + self.b) ) * (-2)
        return db

    def step(self, x, y):
        derivW = np.zeros(9)
        for i in range(9):
            derivW[i] = self.deriv_w(x, y, i)
        derivB = self.deriv_b(x, y)

        self.w = self.w - self.lr * derivW
        self.b = self.b - self.lr * derivB
        return self.w, self.b
    
    def error(self, x, y):
        wx = np.sum(x * self.w)
        e = np.abs(y - (wx + self.b))
        return e

'''
# F2 predict y(PM2.5 now) by x(last t data, with 18 types and t timespan).
# x is a matrix of t*18 number.
'''
class F2:
    def __init__(self, lr, t, alpha):
        self.wp1 = np.ones((18, t)) * 0.1
        self.wp2 = np.ones((18, t)) * 0.1
        self.b = 1
        self.t = t
        self.lr = lr
        self.alpha = alpha
    
    '''
    # all x with power 1 and power 2
    '''

    def deriv(self, x, y): 
        wxp1 = np.sum(self.wp1*x)
        wxp2 = np.sum(self.wp2*np.square(x))
        dwp1 = ( y - (wxp1 + wxp2 + self.b) ) * (-2) * x + 2 * self.alpha * self.wp1
        dwp2 = ( y - (wxp1 + wxp2 + self.b) ) * (-2) * np.square(x) + 2 * self.alpha * self.wp2
        db = ( y - (wxp1 + wxp2 + self.b) ) * (-2)
        # print('dwp1:' + json.dumps(dwp1.tolist()))
        return dwp1, dwp2, db

    def step(self, x, y):
        dwp1, dwp2, db = self.deriv(x, y)

        self.wp1 = self.wp1 - self.lr * dwp1
        self.wp2 = self.wp2 - self.lr * dwp2
        self.b = self.b - self.lr * db
    
    def error(self, x, y):
        wxp1 = np.sum(self.wp1*x)
        wxp2 = np.sum(self.wp2*np.square(x))
        e = np.abs(y - (wxp1 + wxp2 + self.b))
        return e

    def predict(self, x, y):
        wxp1 = np.sum(self.wp1*x)
        wxp2 = np.sum(self.wp2*np.square(x))
        ans = wxp1 + wxp2 + self.b
        return ans

    def parameter(self):
        return self.wp1, self.wp2, self.b


'''
# F3 predict y(PM2.5 now) by x(last t data, with 18 types and t timespan).
# x is a matrix of t*18 number.
'''
class F3:
    def __init__(self, lr, t, alpha):
        self.wp1 = np.ones((18, t)) * 0.01
        self.wp2 = np.ones((18, t)) * 0.001
        self.wp3 = np.ones((18, t)) * 0.0001
        self.b = 1
        self.t = t
        self.lr = lr
        self.alpha = alpha
    
    '''
    # all x with power 1, power 2 and power 3
    '''

    def deriv(self, x, y): 
        wxp1 = np.sum(self.wp1*x)
        wxp2 = np.sum(self.wp2*pow(x, 2))
        wxp3 = np.sum(self.wp3*pow(x, 3))
        dwp1 = ( y - (wxp1 + wxp2 + wxp3 + self.b) ) * (-2) * x + 2 * self.alpha * self.wp1
        dwp2 = ( y - (wxp1 + wxp2 + wxp3 + self.b) ) * (-2) * pow(x, 2) + 2 * self.alpha * self.wp2
        dwp3 = ( y - (wxp1 + wxp2 + wxp3 + self.b) ) * (-2) * pow(x, 3) + 2 * self.alpha * self.wp3
        db = ( y - (wxp1 + wxp2 + wxp3 + self.b) ) * (-2)
        # print('dwp1:' + json.dumps(dwp1.tolist()))
        return dwp1, dwp2, dwp3, db

    def step(self, x, y):
        dwp1, dwp2, dwp3, db = self.deriv(x, y)

        self.wp1 = self.wp1 - self.lr * dwp1
        self.wp2 = self.wp2 - self.lr * dwp2
        self.wp3 = self.wp3 - self.lr * dwp3
        self.b = self.b - self.lr * db
    
    def error(self, x, y):
        wxp1 = np.sum(self.wp1*x)
        wxp2 = np.sum(self.wp2*pow(x, 2))
        wxp3 = np.sum(self.wp3*pow(x, 3))
        e = np.abs(y - (wxp1 + wxp2 + wxp3 + self.b))
        return e

    def predict(self, x, y):
        wxp1 = np.sum(self.wp1*x)
        wxp2 = np.sum(self.wp2*pow(x, 2))
        wxp3 = np.sum(self.wp3*pow(x, 3))
        ans = wxp1 + wxp2 + wxp3 + self.b
        return ans

    def parameter(self):
        return self.wp1, self.wp2, self.wp3, self.b



'''
# F4 predict y(PM2.5 now) by x(last t data, with 18 types and t timespan, and we can set the max power of the function).
# x is a matrix of t*18 number.
'''
class F4:
    def __init__(self, lr, t, maxPower, alpha):
        self.w = np.ones((maxPower, 18, t))
        self.maxPower = maxPower
        self.b = 1
        self.t = t
        self.lr = lr
        self.alpha = alpha
    
    '''
    # all x with power 1, 2, 3 ... maxPower.
    '''

    def deriv(self, x, y): 
        wxSum = 0
        for i in range(self.maxPower):
            wxSum += np.sum(self.w[i]*pow(x, i + 1))
        dw = []
        for i in range(self.maxPower):
            dw.append(( y - (wxSum + self.b) ) * (-2) * pow(x, i + 1) + 2 * self.alpha * self.w[i])
        dw = np.array(dw)
        db = ( y - (wxSum + self.b) ) * (-2)
        # print('dwp1:' + json.dumps(dwp1.tolist()))
        return dw, db

    def step(self, x, y):
        dw, db = self.deriv(x, y)
        for i in range(self.maxPower):
            self.w[i] = self.w[i] - self.lr * dw[i]
        self.b = self.b - self.lr * db
    
    def error(self, x, y):
        wxSum = 0
        for i in range(self.maxPower):
            wxSum += np.sum(self.w[i]*pow(x, i + 1))
        e = np.abs(y - (wxSum + self.b))
        return e

    def predict(self, x):
        wxSum = 0
        for i in range(self.maxPower):
            wxSum += np.sum(self.w[i]*pow(x, i + 1))
        y = wxSum + self.b
        return y

    def parameter(self):
        return self.w, self.b

    def setParameter(self, w, b):
        self.w = w
        self.b = b
