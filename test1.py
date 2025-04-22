import numpy as np
from app import solve_primal, solve_dual, davn_generator, emsr

# Define input data
fare = np.array([
    350, 375, 400, 430, 450, 500,
    600, 610, 620, 630, 640, 650,
    500, 525, 550, 585, 600, 650,
    750, 760, 770, 780, 790, 800
])

demand = np.array([
    58.8, 67.2, 50.4, 58.8, 67.2, 50.4,
    84, 100.8, 84, 75.6, 84, 58.8,
    14.7, 16.8, 12.6, 14.7, 16.8, 12.6,
    21, 25.2, 21, 18.9, 21, 14.7
])

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
print("Optimal booking decisions (x):")
print(x_opt)
print("\nObjective value (Revenue):")
print(obj)

# Run dual optimization
v_opt = solve_dual(fare, demand, capacity, product_to_legs)
print("\nShadow prices (v):")
print(v_opt)

# Generate DAVN bid prices
davn = davn_generator(v_opt, fare, product_to_legs)
print("\nDAVN Bid-Price Matrix:")
print(davn)

# Compute and print EMSR-b booking limits per leg
print("\nBooking Limits by Leg:")
for l in range(len(capacity)):
    leg_index = l + 1
    valid_mask = davn[:, l] != -1
    virtual_fares = davn[valid_mask, l]
    mean_demand = demand[valid_mask]

    # Sort by virtual fare (ascending)
    sorted_indices = np.argsort(virtual_fares)
    vf_sorted = virtual_fares[sorted_indices]
    md_sorted = mean_demand[sorted_indices]

    # Compute booking limits
    bl = emsr(vf_sorted, md_sorted, cancel_prob[l], capacity[l])
    print(f"\nLeg {leg_index}:")
    print("Products:", np.where(valid_mask)[0][sorted_indices] + 1)  # 1-based indexing
    print("Virtual Fares:", vf_sorted.astype(int))
    print("Booking Limits:", bl)
