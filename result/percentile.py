import collections
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:d} ".format(absolute)

a = [74119, 97466, 97891, 93300, 97523, 93839, 92578, 97412, 94556, 96771, 70932, 83624, 96547]

a = np.array(a)
b = np.sort(np.round(a/99247, 2))

freq = {}

freq[0.70] = len(b[np.logical_and(b>=0.7, b<0.9)])
freq[0.90] = len(b[np.logical_and(b>=0.9, b<0.95)])
freq[0.95] = len(b[np.logical_and(b>=0.95, b<=1)])
print(freq)

data = [freq[0.70], freq[0.90], freq[0.95]]

plt.pie(data, labels=['80%-90%', '90%-95%', '95%-100%'], autopct=lambda pct: func(pct, data), textprops=dict(color="black"))
# fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
# ax.pie(data, autopct=lambda pct: func(pct, data),
#                                   textprops=dict(color="w"),labels=['80%-90%', '90%-95%', '95%-100%'])
plt.title('agent performance on test dataset')
plt.savefig('percentile.png')

