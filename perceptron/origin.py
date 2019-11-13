# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:36:49 2019

@author: ransh
"""

import os
os.chdir("E:/exercise_nlp")
import re
import numpy
from re import match

def updateW(w, eta, t):  # w = 0, eta = 1, t = [1, 3, 3]
                        # w = [1, 1], eta = 1, t = [-1, 1, 1]
    y = t[0]
    updateW = []
    index = 0
    if isinstance(w, int):
        for xi in t[1:]:
            updateW.append(w + eta * y * xi)        # updateW = [3, 3]
            index += 1
    else:
        for xi in t[1:]:
            updateW.append(w[index] + eta * y * xi)
            index += 1
    
    return updateW          # updateW = [3, 3]

def updateB(b, eta,t):    # b = 0, eta = 1, t = [1, 3, 3]
    updateB = b + eta * t[0]
    return updateB

def neg(prod):
    flag = 0
    for p in prod[1:]:
        if not match('^-[0-9]*[1-9][0-9]*$', str(p)):
            flag = 1
    if (flag == 0 and prod[0] <= 0):
        return True
    return False

def getProd(sign, y):        # sign = [0, 0, 0], y = 1
                             # sign = [1, 9, 9], y = 1
    prod = []
    for s in sign:
        prod.append(s * y)
    return sum(prod)         # prod = [0, 0, 0]   prod = [1, 9, 9]

def getSign(w, b, x):       # w = 0, b = 0, x = [3, 3]
                            # w = [3, 3], b = 1, x = [3, 3]
                            # w = [3, 3], b = 1, x = [4, 3]
                            # w = [3, 3], b = 1, x = [1, 1]
                            # w = [1, 1], b = 0, x = [1, 1]
    sign = []
    sign.append(b)          # sign = [0]   sign = [1]
    if isinstance(w, int):     # w = 0
        for xi in x:
            sign.append(w * xi)   # sign = [0, 0, 0]
    else:   # w = [3, 3], b = 1, x = [3, 3]
        wx = list(numpy.array(w) * numpy.array(x))
        for wxi in wx:
            sign.append(wxi)
    return sign

def allPerceptron(w0, b0, eta, T):      # 0, 0, 1, T= [[1, 3, 3],
                                        #              [1, 4, 3],
                                        #              [-1, 1, 1]]
    wrong = -1
    w = w0
    b = b0
    while wrong != 0 :
        wrong, w, b, fx = perceptron(wrong, w, b, eta, T)
        print(fx)
    return fx

def perceptron(wrong, w, b, eta, T):       # 0, 0, 1, T = [[1, 3, 3],
                                    #               [1, 4, 3],
                                    #               [-1, 1, 1]]
    fx = []
    flag = 0
    for t in T:     # [1, 3, 3]  [1, 4, 3] [-1, 1, 1]
        sign = getSign(w, b, t[1:])   # [0, 0, 0]
        prod = getProd(sign, t[0])      # [0, 0, 0] [y*b, y*w*xi...] 
        if (prod <= 0):
        #if ((prod[0] <= 0) and (numpy.nonzero(prod[1:])[0].size == 0)) or (neg(prod)):
            w = updateW(w, eta, t)     # w = [3, 3]
            b = updateB(b, eta, t)      # b = 1
            wrong = T.index(t) + 1
            fx.append(w)
            fx.append(b)
            flag = 1
            break
    if(flag == 0):
        wrong = 0
        
    return wrong, w, b, fx
    
def handle(textfile):
    fo = open(textfile, "r")
    print ("文件名为: ", fo.name)
    T = []
    for line in fo.readlines():              # 1 (3,3)          #依次读取每行
        line = line.strip()                  # 1 (3,3)          #去掉每行头尾空白
        data = re.split(r' ', line)          # data = ["1", "(3,3)"]
        t = []
        t.append(int(data[0]))               # t = [1]
        tx = filter(None, re.split(r'[(,)]',data[1]))   # tx = ["3", "3"]
        for txi in tx:
            t.append(int(txi))                  # t = [1, 3, 3]
        T.append(t)                     # T = [[1, 3, 3]]
    fo.close()                      # T = [[1, 3, 3],
                                    #      [1, 4, 3],
                                    #      [-1, 1, 1]]
    return T
        
if __name__ == '__main__':
    T = handle("data/perceptron.txt")
    fx = allPerceptron(0, 0, 1, T)
