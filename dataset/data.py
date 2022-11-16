import numpy
import pandas as pd

def getDataList(file):
    f = open(file,'r')
    a =  []
    for line in f:
        a.append(int(line.strip()))
    train = a[:int(0.8*len(a))]
    test = a[int(0.8*len(a)):]
    return train, test

