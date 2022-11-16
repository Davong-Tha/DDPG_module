from mknapsack import solve_multiple_knapsack

# Given ten items with the following profits and weights:
profits = [18, 9, 23, 20, 59, 61, 70, 75, 76, 30]
weights = [18, 9, 23, 20, 59, 61, 70, 75, 76, 30]

# ...and two knapsacks with the following capacities:
capacities = [sum(weights)/2, sum(weights)/2]

#if remain assign it to the one with smallest capacity overflow

# Assign items into the knapsacks while maximizing profits
res = solve_multiple_knapsack(profits, weights, capacities)
print(res)