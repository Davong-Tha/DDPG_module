import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

a = [1762.0, 1891.0, 2004.0,1680.0, 1948.0, 2051.0,1844.0,1793.0, 1876.0, 1976.0, 1849.0, 1976.0,1855.0, 1907.0, 1722.0, 1827.0, 1833.0, 1793.0,1779.0,1675.0]
b = list(np.array(a)/2367)
avg = sum(a)/len(a)
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
ax.set_xlabel('percentage of instruction completed')
plt.title('task allocation using DDPG')
plt.show()

