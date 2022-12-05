import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

a = [1762.0, 1891.0, 2004.0,1680.0, 1948.0, 2051.0,1844.0,1793.0, 1876.0, 1976.0, 1849.0, 1976.0,1855.0, 1907.0, 1722.0, 1827.0, 1833.0, 1793.0,1779.0,1675.0]
c = [
1715.0,2028.0,2032.0,


1527.0,


1871.0,


1924.0,

2105.0,


1929.0,


1721.0,


2038.0,


1649.0,


2029.0,


2033.0,


49.0,


1878.0,


2017.0,


2222.0,


1385.0,


1817.0,


1883.0,
]

d = [2102,
1867,
1868,
1880,
1869,
1804,
1952,
1864,
2024,
1870,
1889,
1852,
2147,
2054,
1836,
2100,
1861,
1796,
2118,
2146]
b = list(np.array(d)/2367)
avg = sum(d)/len(d)
print(avg)
print(avg/2367)
print(sum(i >= 0.70 and i < 0.80 for i in b))


# plt.plot(range(len(a)), a)
# plt.show()

bins=np.arange(0, 1, 0.1)

fig, ax = plt.subplots(figsize=(10,10))
(
    pd.cut(b, bins=np.arange(0.5, 1, 0.1))
        .value_counts()
        .sort_index()
        .plot.bar(ax=ax)
)
ax.set_ylabel('frequency')
ax.set_xlabel('percentage of tasks completed')
plt.title('task allocation using DDPG')
plt.savefig('eval')


