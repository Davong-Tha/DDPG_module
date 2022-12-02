import matplotlib.pyplot as plt
import numpy as np

def plotLearning(scores, filename, x=None, window=5):
    plt.clf()
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)


def plotconvergence(delay, score, filename):
    plt.clf()
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(range(len(delay)), delay)

    # plt.subplots(122)
    axs[1].plot(range(len(score)), score)


    plt.savefig(filename)