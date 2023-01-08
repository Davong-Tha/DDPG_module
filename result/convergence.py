import matplotlib.pyplot as plt

with open('convergence', 'r') as f:
    result = []
    while f:
        max_delay = float(f.readline())
        completed = float(f.readline())
        total = float(f.readline())
        result.append((completed/total, max_delay))

        if f.readline() == 'end':
            break

result = sorted(result, key=lambda x: x[0])

percent_completed_result = []
max_delay_result = []
for x in result:
    percent_completed_result.append(x[0])
    max_delay_result.append(x[1])

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(range(len(percent_completed_result)),
        percent_completed_result,
        color="red",)
# set x-axis label
ax.set_xlabel("epoch", fontsize = 14)
# set y-axis label
ax.set_ylabel("percentage task completed",
              color="red",
              fontsize=14)


# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
from scipy.signal import savgol_filter
max_delay_result = savgol_filter(max_delay_result, 15, 3)
ax2.plot(range(len(percent_completed_result)),max_delay_result,color="blue")
ax2.set_ylabel("max delay",color="blue",fontsize=14)
plt.show()
# save the plot as a file
fig.savefig('convergence.png',
            dpi=100,
            bbox_inches='tight')

plt.show()