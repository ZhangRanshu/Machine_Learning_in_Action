# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:43:21 2019

@author: ransh
"""
import os
os.chdir("E:/exercise_nlp")
import re
import numpy

def getGram(T):
    N = T.shape[0]
    gram = numpy.zeros((N, N))
    X = T[:,1:N]                # X = [[3 3]
                                #      [4 3]
                                #      [1 1]]
    for i in range(N):
        xi = X[i, :]                    # xi = [3 3]
        for j in range(N):
            xj = X[j, :]                # xj = [3 3]
            gram[i, j] = xi * xj.T      # gram[0, 0] = 18
    return gram                         # [[18. 21.  6.]
                                        #  [21. 25.  7.]
                                        #  [ 6.  7.  2.]]

def getSign(a, y, grami, b):# a = [0 0 0], y = [1 1 -1], grami = [18 21 6]
    sign = 0
    #print(grami)
    for n in range(a.shape[1]):
        sign += a[0, n] * y[n, 0] * grami[n]
        #print(sign)
    return sign + b         # sign = 0

def getProd(sign, yi):  # yi = -1, sign = 0
    return sign * yi

def getA(n, a, eta):
    a[0, n] = a[0, n] + eta
    return a

def getB(b, eta, y):
    return b + eta * y

def perceptron(wrong, a, b, eta, y, gram):  # a = [0 0 0], b = 0, y = [-1 1 1]
                                 # gram = [[18. 21.  6.]
                                #          [21. 25.  7.]
                                #          [ 6.  7.  2.]]
    flag = 0
    for n in range(a.shape[1]):
        grami = gram[:, n]      # grami = [18 21 6]
        sign = getSign(a, y, grami, b)  # sign = 0
        prod = getProd(sign, y[n])
        if prod <= 0 :
            a = getA(n, a, eta)
            b = getB(b, eta, y[n])
            flag = 1
            wrong = n + 1
            break
    if(flag == 0):
        wrong = 0

    return wrong, a, b
    
def allPerceptron(eta, T):      # 1, T= [[1 3 3]
                                #        [1 4 3]
                                #        [-1 1 1]]
    a = numpy.mat([0] * T.shape[0])        # a = [0 0 0]
    b = 0
    y = T[:, 0]                 # y = [1 1 -1]
    gram = getGram(T)           # [[18. 21.  6.]
                                #  [21. 25.  7.]
                                #  [ 6.  7.  2.]]
    wrong = -1
    while wrong != 0 :
        wrong, a, b = perceptron(wrong, a, b, eta, y, gram)
        print("a, b = ")
        print(a)
        print(b)
        print("---------------")
        #wrong = 0

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
    fo.close()                      
    return numpy.mat(T)             # T = [[1 3 3]
                                    #      [1 4 3]
                                    #      [-1 1 1]]

if __name__ == '__main__':
    T = handle("data/perceptron.txt")
    allPerceptron(1, T)