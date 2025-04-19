# streamlit_app_pulp_detailed.py

import streamlit as st
import numpy as np
import pandas as pd
import pulp
from scipy.stats import poisson
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔍 DAVN & Per‑Leg EMSR‑b – Detailed Debug View")

# --- Helpers & Caching ---
@st.cache_data
def leg_finder(product_idx, product_to_legs):
    legs = product_to_legs[product_idx - 1]
    return [leg for leg in legs if leg > 0]

@st.cache_data
def emsr(fares, demand, cancel_prob, capacity):
    # EMSR‑b (emsr.m) :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
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

@st.cache_data
def solve_primal(fare, demand, capacity, product_to_legs):
    # Primal LP: max  fare⋅x  s.t.  x≤demand,  A x ≤ capacity
    P, L = len(fare), len(capacity)
    prob = pulp.LpProblem("PrimalDAVN", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(P), lowBound=0)
    # objective
    prob += pulp.lpSum(fare[j] * x[j] for j in range(P))
    # x_j ≤ demand_j
    for j in range(P):
        prob += x[j] <= demand[j]
    # leg capacities
    for l in range(1, L+1):
        prods = [j for j in range(P) if l in leg_finder(j+1, product_to_legs)]
        prob += pulp.lpSum(x[j] for j in prods) <= capacity[l-1]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    x_opt = np.array([x[j].value() for j in range(P)], dtype=float)
    obj = pulp.value(prob.objective)
    return x_opt, obj

@st.cache_data
def solve_dual(fare, demand, capacity, product_to_legs):
    # Dual LP: min demand⋅u + capacity⋅v  s.t. u_j + ∑_{l∈legs_j}v_l ≥ fare_j
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

@st.cache_data
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

# --- Step 1: Solve Primal & Dual ---
st.header("1️⃣ Master LP Solutions")

x_opt, obj = solve_primal(fare, demand, capacity, product_to_legs)
v_opt = solve_dual(fare, demand, capacity, product_to_legs)

st.subheader("Primal booking x (Z₀ to Z₂₃) and upper‐bound profit")
df_x = pd.DataFrame({
    "Product": [f"Z{j}" for j in range(P)],
    "xᵢ": np.round(x_opt, 4),
    "fareᵢ": fare
})
st.dataframe(df_x, use_container_width=True)
st.markdown(f"**Upper bound on profit from LP**: {obj:.0f}")

st.subheader("Dual shadow prices per leg")
df_v = pd.DataFrame({
    "Leg": [f"L{l}" for l in range(1, L+1)],
    "shadow_price": v_opt.astype(int)
})
st.dataframe(df_v, use_container_width=True)

# bar chart of shadow prices
fig1, ax1 = plt.subplots()
ax1.bar(df_v["Leg"], df_v["shadow_price"])
ax1.set_title("Shadow Prices")
ax1.set_ylabel("Value")
st.pyplot(fig1)

# --- Step 2: DAVN Bid‑Price Matrix ---
st.header("2️⃣ DAVN Bid‑Price Matrix")

davn = davn_generator(v_opt, fare, product_to_legs)
df_davn = pd.DataFrame(
    davn,
    index=[f"P{p}" for p in range(1, P+1)],
    columns=[f"L{l}" for l in range(1, L+1)]
)
st.dataframe(df_davn, use_container_width=True)

# heatmap
fig2, ax2 = plt.subplots(figsize=(6, 6))
cax = ax2.imshow(df_davn.values, aspect="auto")
ax2.set_xticks(range(L));   ax2.set_xticklabels(df_davn.columns, rotation=45)
ax2.set_yticks(range(P));   ax2.set_yticklabels(df_davn.index)
ax2.set_title("DAVN Bid‑Price Heatmap")
fig2.colorbar(cax, ax=ax2, label="Bid price")
st.pyplot(fig2)

# --- Step 3: Intermediate Matrices to Match Your Printout ---
st.header("3️⃣ Intermediate Matrices (match your debug)")

# products per leg
leg_products = {
    l: sorted([p for p in range(1, P+1) if l in leg_finder(p, product_to_legs)])
    for l in range(1, L+1)
}

# build 6×6 matrices
rows = 6
leg_pr_mat = pd.DataFrame(
    {f"L{l}": pd.Series(leg_products[l], index=range(rows)) 
     for l in range(1, L+1)}
).astype('Int64')
leg_fare_mat = pd.DataFrame({
    f"L{l}": pd.Series([davn[p-1, l-1] for p in leg_products[l]], index=range(rows))
    for l in range(1, L+1)
})
mean_dem_mat = pd.DataFrame({
    f"L{l}": pd.Series([demand[p-1] for p in leg_products[l]], index=range(rows))
    for l in range(1, L+1)
})

st.subheader("leg_product")
st.table(leg_pr_mat)

st.subheader("leg_fare")
st.table(leg_fare_mat)

st.subheader("mean_demand")
st.table(mean_dem_mat)

# sorted matrices
sorted_fare = leg_fare_mat.apply(np.sort)
idx0 = leg_fare_mat.apply(lambda col: np.argsort(col.values), axis=0)
idx1 = idx0 + 1  # to 1‑based
sorted_dem = pd.DataFrame({
    f"L{l}": sorted(mean_dem_mat[f"L{l}"].values)
    for l in range(1, L+1)
})

st.subheader("reordered_leg_fare")
st.table(sorted_fare)

st.subheader("index (1‑based)")
st.table(idx1)

st.subheader("reordered_demand")
st.table(sorted_dem)

# --- Step 4: Per‑Leg EMSR‑b & Virtual Fares + BL  (match your BL printout) ---
st.header("4️⃣ Per‑Leg EMSR‑b Booking Limits (with Virtual Fares)")

for l in range(1, L+1):
    prods = leg_products[l]
    vf = sorted_fare[f"L{l}"].values
    md = sorted_dem[f"L{l}"].values
    bl = emsr(vf, md, cancel_prob[l-1], capacity[l-1])
    st.subheader(f"leg = {l}")
    st.code(f"products_on_leg = {prods}", language="matlab")
    st.code(f"VIRTUAL_FARES =\n{vf.astype(int)}", language="matlab")
    st.code(f"BL =\n{bl}", language="matlab")
