# test_optimization.py

import numpy as np
from app import solve_primal, solve_dual, davn_generator, emsr

# Define input data
fare = np.array([350, 375, 400, 430, 450, 500, 600, 610, 620, 630, 640, 650,
                 500, 525, 550, 585, 600, 650, 750, 760, 770, 780, 790, 800])
demand = np.array([58.8, 67.2, 50.4, 58.8, 67.2, 50.4,
                   84, 100.8, 84, 75.6, 84, 58.8,
                   14.7, 16.8, 12.6, 14.7, 16.8, 12.6,
                   21, 25.2, 21, 18.9, 21, 14.7])
capacity = np.array([100, 100, 100, 100, 100, 100])
cancel_prob = np.array([0.225, 0.2, 0.1, 0.22, 0.15, 0.21])
product_to_legs = np.array([
    [1, -1], [2, -1], [3, -1], [4, -1], [5, -1], [6, -1],
    [2, 3], [1, 4], [2, 5], [1, 6], [4, 5], [3, 6],
    [1, -1], [2, -1], [3, -1], [4, -1], [5, -1], [6, -1],
    [2, 3], [1, 4], [2, 5], [1, 6], [4, 5], [3, 6]
])

# Run primal optimization
x_opt, obj = solve_primal(fare, demand, capacity, product_to_legs)
print("Optimal booking decisions (x):", x_opt)
print("Objective value (Revenue):", obj)

# Run dual optimization
v_opt = solve_dual(fare, demand, capacity, product_to_legs)
print("Shadow prices (v):", v_opt)

# Generate DAVN bid prices
davn = davn_generator(v_opt, fare, product_to_legs)
print("DAVN Bid-Price Matrix:")
print(davn)

# Test EMSR-b booking limits
for l in range(len(capacity)):
    vf = davn[:, l][davn[:, l] != -1]  # Filter out invalid values (-1)
    md = demand[davn[:, l] != -1]  # Match demand for valid products
    bl = emsr(vf, md, cancel_prob[l], capacity[l])
    print(f"Booking limits for Leg {l+1}:", bl)