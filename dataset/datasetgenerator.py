import csv
import numpy as np
import random


class DatasetGenerator:
    def __init__(self, average_load, num_nodes, exp_param):
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng()
        self.average_load = average_load
        self.cpu_power = [self.rng.normal(exp_param) for i in range(num_nodes)]

    def get_random_task(self):
        # n = self.rng.guassian(self.average_load)
        n = self.rng.poisson(self.average_load)
        parts = random.randint(1, n)
        result = []
        for i in range(parts - 1):
            mean = self.average_load // (self.num_nodes * 0.9)
            x = self.rng.poisson(mean)
            if x == 0:
                continue
            # x = random.randint(1, n - parts + i + 1)
            if n > x:
                n = n - x
            else:
                result.append(n)
                break
            result.append(x)


        return result

    def generate_cost_matrix(self, tasks, l_max=1):
        """

        Default case: l_max=1, the cost matrix is built as follows
           for each node i, take the k subjobs (an array) and divide each part by c_i (CPU Power of each node).
        l_max > 1, the cost matrix is built as follows
           for each node i, take the k subjobs (an array) and divide each part by c_i (CPU Power of each node) & CREATE l_max copies.
            Thus, final size is l_max * node
        """
        cost_matrix = []
        # tasks = self.get_random_task()
        for i in range(self.num_nodes):
            for j in range(l_max):
              cost_matrix.append([task / self.cpu_power[i] for task in tasks])
        return cost_matrix


if __name__ == '__main__':
    d = DatasetGenerator(500,  5, 500)
    data =[]
    for i in range(1000):
        data.append(d.get_random_task())

    avg = sum(sum(data,[]))/sum(len(x) for x in data)
    d.cpu_power = [int(np.random.default_rng().normal(avg, 15)) for i in range(d.num_nodes)]

    with open('dataset1000.csv', 'w', newline='') as f:
        write = csv.writer(f)
        # using csv.writer method from CSV package
        write.writerow(list(d.cpu_power))
        write.writerows(data)
