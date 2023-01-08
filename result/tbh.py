import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# creating the dataset
data = {'10 nodes': [65,70,75],
        '20 nodes': [75,80,85],
        '30 nodes': [85,90,95]}

df = pd.DataFrame(data,columns=['10 nodes', '20 nodes', '30 nodes'], index = ['500', '1000', '1500'])

# Multiple bar chart

df.plot(kind='bar', rot=0)

# Display
plt.xlabel('average task size', size=20)
plt.ylabel('% task completed', size=20)
plt.savefig('different node and task size.png')


plt.show()