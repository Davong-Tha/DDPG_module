import random

import numpy as np
import pandas as pd

def getDataList(file):
    f = open(file,'r')
    a =  []
    for line in f:
        a.append(int(line.strip()))
    # train = a[:int(0.8*len(a))]
    # test = a[int(0.8*len(a)):]
    train = []
    test = []
    for data in a[:int(0.8*len(a))]:
        size = random.randint(4, 10)
        train.append(list(np.ceil(data * np.random.dirichlet([1]*size))))
    for data in a[int(0.8*len(a)):]:
        size = random.randint(4, 10)
        test.append(list(np.ceil(data * np.random.dirichlet([1] * size))))
    return train, test

if __name__ == '__main__':
    train, test = getDataList('100dataset.txt')
    print(train)

