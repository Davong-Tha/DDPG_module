import numpy as np
import cvxpy as cvx
from cvxpy import SCS

# N = # workers
# M = # jobs
N = 4
M = 5

# generate data
task_sizes = np.random.normal(10, 1, size=(1, M)) # link
cpu_powers = np.random.normal(5, 1, (1, N)) # link
# deadline by worker
# deadlines_by_worker = np.random.normal(2, 0, (1, N))
# deadline by task
deadlines_by_task = np.random.normal(10, 0, (1, M)) # link
deadlines_by_task = np.tile(deadlines_by_task, (N, 1))
max_loads = np.random.normal(20, 1, (1, N))

print("parameters")
print("\ntask sizes:\n", task_sizes)
print("\ncpu powers:\n", cpu_powers)
# print("\ndeadlines by worker:\n", deadlines_by_worker)
print("\ndeadlines by task:\n", deadlines_by_task)

# define problem
# variables
allocation_matrix = cvx.Variable((M, N), boolean=True)

# constaints
constraints = []
# assign one task to one worker only
constraints += [ cvx.sum(allocation_matrix, axis=1) == np.ones(M) ]
#  max load constraint
constraints += [ max_loads - task_sizes @ allocation_matrix >= np.zeros((1, N)) ]

# objective
# get minimum deadline of tasks assigned to each worker
overall_deadlines = cvx.reshape( cvx.min( cvx.multiply( deadlines_by_task.T, allocation_matrix ), axis=0 ), (1, N) )

# objective = cvx.sum( cvx.maximum( ( cvx.multiply( task_sizes @ allocation_matrix, 1 / cpu_powers ) - deadlines_by_worker ), 0) )
objective = cvx.sum( cvx.maximum( ( cvx.multiply( task_sizes @ allocation_matrix, 1 / cpu_powers ) - overall_deadlines ), 0) )

prob = cvx.Problem(cvx.Minimize(objective), constraints)
prob.solve()


print('\noptimal assignment:\n', allocation_matrix.value)
print('\noptimal value:\n', prob.value)
print(prob.status)