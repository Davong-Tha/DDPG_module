import csv
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

def getDataFromCSV(file):
    f = open(file, 'r')
    reader = csv.reader(f)
    data = list(reader)
    cpu = [int(x) for x in data[0]]
    data = data[1:]
    data = [list(map(int,i)) for i in data]
    train, test= train_test_split(data, test_size=0.2)
    # print(sum(sum(data,[]))/sum(len(x) for x in data))
    return cpu, train, test

if __name__ == '__main__':
    # train, test = getDataList('100dataset.txt')
    # print(train)
    print(getDataFromCSV('dataset1000.csv'))

