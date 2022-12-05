# import numpy as np
# from mknapsack import solve_multiple_knapsack
#
# from dataset import data
#
#
#
#
# def allocateTask(sort_train, predicted_load_capacity):
#     allocation = [0] * predicted_load_capacity
#     for a in sort_train:
#         assign = False
#         for i in range(len(predicted_load_capacity)):
#             if a < predicted_load_capacity[i] - allocation[i]:
#                 allocation[i] += a
#                 assign = True
#                 break
#
#         if assign:
#             continue
#
#         exceed = []
#         for i in range(len(predicted_load_capacity)):
#             exceed.append(a - predicted_load_capacity[i] + allocation[i])
#         best_exceed = np.argmin(np.array(exceed))
#         allocation[best_exceed] += a
#     return allocation
#
#
# if __name__ == '__main__':
#     predicted_load_capacity = np.ceil(np.random.normal([5,10,15], [5, 5, 5], (3, 3)))
#     train, test = data.getDataList('100dataset.txt')
#     CPUPower = np.array([5, 10, 15])
#     ddl = 5
#     print(predicted_load_capacity[0])
#     print(train[0])
#
#
#     sort_train = sorted(train[0], reverse=True)
#     allocation = np.array(allocateTask(sort_train, predicted_load_capacity[0]))
#     state = allocation/CPUPower < ddl
#     reward = state * allocation
#     print(allocation)
#     print(sum(allocation) == sum(sort_train))
#     print('reward', reward)