# streamlit_app_pulp_detailed.py

import streamlit as st
import numpy as np
import pandas as pd
import pulp
from scipy.stats import poisson
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("DAVN & EMSR-b Airline Revenue Management")

# --- Core Functions ---
def leg_finder(product_idx, product_to_legs):
    legs = product_to_legs[product_idx - 1]
    return [leg for leg in legs if leg > 0]

def emsr(fares, demand, cancel_prob, capacity):
    n = len(fares)
    agg_d = np.zeros(n)
    agg_f = np.zeros(n)
    total_d = total_wf = 0.0
    for i in range(n):
        j = n - 1 - i
        total_d += demand[j]
        total_wf += fares[j] * demand[j]
        agg_d[j] = total_d
        agg_f[j] = total_wf / total_d
    prot = np.zeros(n)
    for i in range(1, n):
        frac = (agg_f[i] - fares[i - 1]) / agg_f[i]
        prot[i] = poisson.ppf(frac, agg_d[i])
    prot[0] = 0.0
    C = capacity / (1 - cancel_prob)
    bl = np.maximum(C - prot, 0)
    bl[-1] = C
    return np.ceil(bl).astype(int)

def solve_primal(fare, demand, capacity, product_to_legs):
    P, L = len(fare), len(capacity)
    prob = pulp.LpProblem("PrimalDAVN", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(P), lowBound=0)
    prob += pulp.lpSum(fare[j] * x[j] for j in range(P))
    for j in range(P):
        prob += x[j] <= demand[j]
    for l in range(1, L+1):
        prods = [j for j in range(P) if l in leg_finder(j+1, product_to_legs)]
        prob += pulp.lpSum(x[j] for j in prods) <= capacity[l-1]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    x_opt = np.array([x[j].value() for j in range(P)], dtype=float)
    obj = pulp.value(prob.objective)
    return x_opt, obj

def solve_dual(fare, demand, capacity, product_to_legs):
    P, L = len(fare), len(capacity)
    prob = pulp.LpProblem("DualDAVN", pulp.LpMinimize)
    u = pulp.LpVariable.dicts("u", range(P), lowBound=0)
    v = pulp.LpVariable.dicts("v", range(L), lowBound=0)
    prob += pulp.lpSum(demand[j]*u[j] for j in range(P)) + \
            pulp.lpSum(capacity[l]*v[l] for l in range(L))
    for j in range(P):
        legs = leg_finder(j+1, product_to_legs)
        prob += u[j] + pulp.lpSum(v[l-1] for l in legs) >= fare[j]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    v_opt = np.array([v[l].value() for l in range(L)], dtype=float)
    return v_opt

def davn_generator(shadow_prices, fare, product_to_legs):
    P, L = len(fare), len(shadow_prices)
    davn = -np.ones((P, L))
    for p in range(1, P+1):
        legs = leg_finder(p, product_to_legs)
        ψ = sum(shadow_prices[l-1] for l in legs)
        for l in legs:
            davn[p-1, l-1] = fare[p-1] - ψ + shadow_prices[l-1]
    return davn

# --- Data from your .m files ---
fare = np.array([350,375,400,430,450,500,600,610,620,630,640,650,
                 500,525,550,585,600,650,750,760,770,780,790,800])
demand = np.array([58.8,67.2,50.4,58.8,67.2,50.4,
                   84,100.8,84,75.6,84,58.8,
                   14.7,16.8,12.6,14.7,16.8,12.6,
                   21,25.2,21,18.9,21,14.7])
capacity = np.array([100,100,100,100,100,100])
cancel_prob = np.array([0.225,0.2,0.1,0.22,0.15,0.21])
product_to_legs = np.array([
    [1,-1],[2,-1],[3,-1],[4,-1],[5,-1],[6,-1],
    [2,3],[1,4],[2,5],[1,6],[4,5],[3,6],
    [1,-1],[2,-1],[3,-1],[4,-1],[5,-1],[6,-1],
    [2,3],[1,4],[2,5],[1,6],[4,5],[3,6]
])

P, L = len(fare), len(capacity)

# --- Main App with Run Button ---
if st.button("Run Optimization", type="primary"):
    # Solve LP
    x_opt, obj = solve_primal(fare, demand, capacity, product_to_legs)
    v_opt = solve_dual(fare, demand, capacity, product_to_legs)
    
    # Generate DAVN bid prices
    davn = davn_generator(v_opt, fare, product_to_legs)
    
    # Display DAVN Matrix
    st.header("DAVN Bid-Price Matrix")
    df_davn = pd.DataFrame(
        davn,
        index=[f"P{p}" for p in range(1, P+1)],
        columns=[f"L{l}" for l in range(1, L+1)]
    )
    st.dataframe(df_davn.round(0), use_container_width=True)
    
    # Identify products on each leg
    leg_products = {
        l: sorted([p for p in range(1, P+1) if l in leg_finder(p, product_to_legs)])
        for l in range(1, L+1)
    }
    
    # Calculate sorted fares and demands for each leg
    leg_fare_dict = {}
    leg_demand_dict = {}
    
    for l in range(1, L+1):
        prods = leg_products[l]
        leg_fare_dict[l] = [davn[p-1, l-1] for p in prods]
        leg_demand_dict[l] = [demand[p-1] for p in prods]
        
        # Sort by fare
        sorted_indices = np.argsort(leg_fare_dict[l])
        leg_fare_dict[l] = np.array(leg_fare_dict[l])[sorted_indices]
        leg_demand_dict[l] = np.array(leg_demand_dict[l])[sorted_indices]
        leg_products[l] = np.array(leg_products[l])[sorted_indices]
    
    # Calculate booking limits for each leg and create the combined table
    st.header("Booking Limits by Leg")
    
    # Prepare data for the combined table
    all_rows = []
    
    for l in range(1, L+1):
        vf = leg_fare_dict[l]
        md = leg_demand_dict[l]
        bl = emsr(vf, md, cancel_prob[l-1], capacity[l-1])
        
        for i, p in enumerate(leg_products[l]):
            all_rows.append({
                "Leg": l,
                "Product": p,
                "Virtual Fare": int(vf[i]),
                "Booking Limit": bl[i]
            })
    
    # Create and display table
    booking_table = pd.DataFrame(all_rows)
    st.dataframe(booking_table, use_container_width=True)
    
    # Display profit information
    st.markdown(f"### Revenue Information")
    st.markdown(f"**Upper bound on profit from LP**: ${obj:.0f}")
else:
    st.info("Click the 'Run Optimization' button to calculate DAVN bid prices and booking limits.")
