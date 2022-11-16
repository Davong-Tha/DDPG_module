import numpy as np
import random

np.random.seed(0)

class DatasetGenerator:
    def __init__(self, average_load, num_nodes):
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng()
        self.average_load = average_load

    def get_random_task(self):
        n = self.rng.poisson(self.average_load)
        print('n', n)
        parts = random.randint(1, n)
        print('p', parts)
        result = []
        for i in range(parts - 1):
            mean = self.average_load//(self.num_nodes*0.9)
            x = self.rng.poisson(mean)
            if x == 0:
                continue
            # x = random.randint(1, n - parts + i + 1)
            if n >= x:
                n = n - x
            else:
                result.append(n)
                break

            result.append(x)
        result.append(n)
        print(len(result))
        return result


if __name__ == '__main__':
    d = DatasetGenerator(900,  60)
    for i in range(20):
        print(d.get_random_task())
