import numpy as np
import scipy

from dataset.datasetgenerator import DatasetGenerator

class HungarianMethod:

    def final_assignments(self, row_ind, col_ind, num_nodes, l_max):
      assignments = [[] for _ in range(num_nodes)]
      for i, row_i in enumerate(row_ind):
        node = row_i // l_max

        assignments[node].append(col_ind[i])
      return assignments

    def allocate(self, tasks, num_nodes, cpu_power, l_max=2):
        cost_matrix = DatasetGenerator.generate_cost_matrix(tasks, num_nodes, cpu_power, l_max)
        # see that the cost_matrix has shape (num_nodes * l_max) x num_tasks
        costs = np.array(cost_matrix)
       # then perform optimization for job assignments using hungarian method
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(costs)
        return self.final_assignments(row_ind, col_ind, num_nodes, l_max)