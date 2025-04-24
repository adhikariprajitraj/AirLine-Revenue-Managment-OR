# streamlit_app_pulp_detailed.py

import streamlit as st
import numpy as np
import pandas as pd
import pulp
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

def leg_finder(product_idx, product_to_legs):
    legs = product_to_legs[product_idx - 1]
    return [leg for leg in legs if leg > 0]

def emsr(fares, demand, cancel_prob, capacity):
    # Sort by virtual fare just in case
    order = np.argsort(fares)
    fares = np.array(fares)[order]
    demand = np.array(demand)[order]

    n = len(fares)
    # 1) Build cumulative demand & weighted fares
    agg_d = np.zeros(n)
    agg_f = np.zeros(n)
    tot_d = tot_wf = 0.0
    for i in range(n):
        j = n - 1 - i
        tot_d  += demand[j]
        tot_wf += fares[j] * demand[j]
        agg_d[j] = tot_d
        agg_f[j] = tot_wf / tot_d if tot_d > 0 else 0

    # 2) Compute protection levels Q_{i+1} in prot[i+1]
    prot = np.zeros(n + 1)  # Add one extra element for proper indexing
    for i in range(1, n):
        if agg_f[i] > fares[i - 1]:
            frac = (agg_f[i] - fares[i-1]) / agg_f[i]
            prot[i] = poisson.ppf(frac, agg_d[i])
    # prot[0] and prot[n] are unused

    # 3) Adjust capacity for cancellations
    C_eff = capacity / (1 - cancel_prob)

    # 4) Booking limits: correctly shift protection by one
    bl = np.zeros(n)
    
    # For the lowest fare class - it should get C - protection level for next class
    # which is prot[1] for the lowest class
    if n > 0:
        bl[0] = np.maximum(C_eff - prot[1], 0)
    
    # For classes 2...n-1
    for i in range(1, n-1):
        bl[i] = np.maximum(C_eff - prot[i+1], 0)
        
    # Highest fare has full capacity
    if n > 0:
        bl[n-1] = C_eff

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
        psi = sum(shadow_prices[l-1] for l in legs)
        for l in legs:
            # Calculate virtual fare and ensure it's positive
            virtual_fare = fare[p-1] - psi + shadow_prices[l-1]
            davn[p-1, l-1] = max(virtual_fare, 0)  # Ensure non-negative value
    return davn

def plot_davn_heatmap(davn_data):
    # Create a copy to avoid modifying the original
    davn_copy = davn_data.copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a mask for the -1 values
    mask = davn_copy == -1
    
    # Create heatmap with custom colormap
    sns.heatmap(
        davn_copy, 
        annot=True, 
        fmt='.0f',
        mask=mask,
        cmap='viridis',
        linewidths=.5, 
        ax=ax,
        cbar_kws={'label': 'Virtual Fare ($)'}
    )
    
    ax.set_title('DAVN Bid-Price Matrix', fontsize=16)
    plt.tight_layout()
    
    return fig

def plot_booking_limits(booking_data):
    legs = sorted(booking_data['Leg'].unique())
    
    if len(legs) == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        axs = [ax]  # Put the single axis in a list for consistent handling
    else:
        fig, axs = plt.subplots(len(legs), 1, figsize=(10, 4*len(legs)))
        if len(legs) > 1:
            axs = axs.flatten()  # Ensure axs is always an array
    
    for i, leg in enumerate(legs):
        leg_data = booking_data[booking_data['Leg'] == leg].copy()
        
        # Sort by Virtual Fare
        leg_data = leg_data.sort_values('Virtual Fare')
        
        # Create the bar chart
        bars = axs[i].bar(
            ["P" + str(p) for p in leg_data['Product']],
            leg_data['Booking Limit'],
            color='skyblue'
        )
        
        # Add virtual fare labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            axs[i].text(
                bar.get_x() + bar.get_width()/2.,
                height + 5,
                f"${leg_data['Virtual Fare'].iloc[j]}",
                ha='center', va='bottom',
                rotation=0,
                fontsize=9
            )
        
        axs[i].set_title(f'Leg {leg} Booking Limits', fontsize=14)
        axs[i].set_xlabel('Products')
        axs[i].set_ylabel('Booking Limit')
        axs[i].grid(axis='y', alpha=0.3)
        
        # Add capacity line
        capacity_value = capacity[leg-1]
        axs[i].axhline(y=capacity_value, color='r', linestyle='-', alpha=0.7)
        axs[i].text(0, capacity_value+2, f"Capacity: {capacity_value}", color='r')
    
    plt.tight_layout()
    return fig

# --- Default Data values ---
default_fare = np.array([350,375,400,430,450,500,600,610,620,630,640,650,
                 500,525,550,585,600,650,750,760,770,780,790,800])
default_demand = np.array([58.8,67.2,50.4,58.8,67.2,50.4,
                   84,100.8,84,75.6,84,58.8,
                   14.7,16.8,12.6,14.7,16.8,12.6,
                   21,25.2,21,18.9,21,14.7])
default_capacity = np.array([100,100,100,100,100,100])
default_cancel_prob = np.array([0.225,0.2,0.1,0.22,0.15,0.21])
default_product_to_legs = np.array([
    [1,-1],[2,-1],[3,-1],[4,-1],[5,-1],[6,-1],
    [2,3],[1,4],[2,5],[1,6],[4,5],[3,6],
    [1,-1],[2,-1],[3,-1],[4,-1],[5,-1],[6,-1],
    [2,3],[1,4],[2,5],[1,6],[4,5],[3,6]
])

if __name__ == "__main__":
    # Set up session state to track app state
    if "optimization_run" not in st.session_state:
        st.session_state.optimization_run = False
    if "show_viz" not in st.session_state:
        st.session_state.show_viz = False
    if "show_input" not in st.session_state:
        st.session_state.show_input = False
        
    # Initialize data in session state if not present
    if "fare" not in st.session_state:
        st.session_state.fare = default_fare.copy()
    if "demand" not in st.session_state:
        st.session_state.demand = default_demand.copy()
    if "capacity" not in st.session_state:
        st.session_state.capacity = default_capacity.copy()
    if "cancel_prob" not in st.session_state:
        st.session_state.cancel_prob = default_cancel_prob.copy()
    if "product_to_legs" not in st.session_state:
        st.session_state.product_to_legs = default_product_to_legs.copy()

    def run_optimization():
        st.session_state.optimization_run = True
        st.session_state.show_viz = False

    def show_visualizations():
        st.session_state.show_viz = True
        
    def toggle_input_data():
        st.session_state.show_input = not st.session_state.show_input
        
    def reset_to_defaults():
        st.session_state.fare = default_fare.copy()
        st.session_state.demand = default_demand.copy()
        st.session_state.capacity = default_capacity.copy()
        st.session_state.cancel_prob = default_cancel_prob.copy()
        st.session_state.product_to_legs = default_product_to_legs.copy()
        st.success("Data reset to default values!")

    st.set_page_config(layout="wide")
    st.title("DAVN & EMSR-b Airline Revenue Management")

    # --- Description ---
    st.markdown("""
    ## Optimizing Seat Allocation to Maximize Airline Revenue

    Airlines face a daily challenge: how to best manage limited seat capacity on their flights while
    maximizing revenue. The key lies in offering the right seats, at the right price, to the right customers
    — and knowing when to say no to low-fare bookings in hopes of selling those seats at a higher price
    later.

    This is the essence of airline revenue management. Airlines don't just sell one type of ticket — they
    offer multiple fare classes for the same route, ranging from economy saver fares to flexible business
    class tickets. Each fare class has different prices and conditions, and customer demand for these fares
    fluctuates over time.

    To make the most out of every flight, airlines use smart optimization techniques to decide:
    - How many seats should be available at each fare level?
    - When should a cheaper fare stop being offered?
    - How can the value of a seat be assessed when multiple itineraries overlap across a network
    of flights?

    The EMSR-b rule helps airlines set limits on how many tickets to sell at each price level. It uses
    historical demand data to strike a balance: selling seats early at a lower price versus waiting to sell
    them later at a higher price.

    However, in real-world airline operations, things are more complicated. Many customers book trips
    that involve multiple connecting flights, and a seat on one leg of a journey may be shared across
    many different itineraries.

    To handle this, our approach uses a method called Displacement Adjusted Virtual Nesting (DAVN).
    This method helps airlines make smarter decisions by considering the entire flight network. It
    estimates the true revenue value of each ticket, accounting for the opportunity cost of assigning a
    seat to one itinerary instead of another.

    Using a combination of:
    - Linear programming to optimize revenue across the network, and
    - Seat-allocation principles based on pricing and demand to manage booking decisions at the
    individual flight level,

    ...this integrated system enables airlines to manage bookings in a way that is both strategically
    optimal and operationally practical.
    """)
    
    # Display the image using Streamlit's image display instead of markdown
    try:
        # Check if file exists
        image_path = "figure.png"
        if os.path.exists(image_path):
            st.image(image_path, caption="A figure showing a network of four cities where Chicago serves as the hub and Denver, Miami, and Boston serve as the spokes")
        else:
            st.error(f"Image file not found at: {image_path}")
            
            # Option to create the image - this will display if the image is missing
            st.warning("The hub and spoke network image is missing. Would you like to create a simple visualization?")
            if st.button("Generate Hub and Spoke Network Visualization"):
                # Create a simple hub and spoke network visualization
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Define city positions (x, y)
                cities = {
                    "Chicago": (0, 0),
                    "Denver": (-2, 1),
                    "Miami": (1, -2),
                    "Boston": (2, 1.5)
                }
                
                # Plot cities as nodes
                for city, pos in cities.items():
                    ax.plot(pos[0], pos[1], 'o', markersize=15, color='blue' if city == "Chicago" else 'green')
                    ax.text(pos[0], pos[1]+0.2, city, ha='center', fontsize=12, fontweight='bold')
                
                # Draw connections (spokes)
                chicago = cities["Chicago"]
                for city, pos in cities.items():
                    if city != "Chicago":
                        ax.plot([chicago[0], pos[0]], [chicago[1], pos[1]], 'r-', linewidth=2)
                
                ax.set_title("Hub and Spoke Network", fontsize=16)
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 2)
                ax.axis('off')
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(image_path)
                
                # Display the newly created image
                st.image(image_path, caption="A figure showing a network of four cities where Chicago serves as the hub and Denver, Miami, and Boston serve as the spokes")
                st.success("Network diagram created successfully!")
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
    
    st.markdown("""
    ### The Big Picture

    This solution gives airlines a powerful tool to improve profitability without adding more flights or
    seats. By making informed booking decisions backed by optimization models, airlines can increase
    revenue, manage uncertainty in demand, and better utilize their limited capacity.
    """)

    # Add button to show/hide input data with toggle_input_data callback
    st.button("Edit Input Data", on_click=toggle_input_data)
    
    # Get current values from session state for convenience
    fare = st.session_state.fare
    demand = st.session_state.demand
    capacity = st.session_state.capacity
    cancel_prob = st.session_state.cancel_prob
    product_to_legs = st.session_state.product_to_legs
    
    P, L = len(fare), len(capacity)
    
    # Display and edit input data if toggled
    if st.session_state.show_input:
        st.header("Input Data Editor")
        st.info("Edit the values directly in the tables below. Changes will be used in the next optimization run.")
        
        # Add reset button
        st.button("Reset to Default Values", on_click=reset_to_defaults)
        
        tab1, tab2, tab3 = st.tabs(["Products", "Legs", "Network"])
        
        with tab1:
            # Create editable product data
            product_data = pd.DataFrame({
                'Product': [f'P{i+1}' for i in range(P)],
                'Fare ($)': st.session_state.fare,  # Use session state directly
                'Expected Demand': st.session_state.demand  # Use session state directly
            })
            
            st.subheader("Product Fares and Demand")
            edited_product_data = st.data_editor(
                product_data,
                use_container_width=True,
                hide_index=True,
                disabled=["Product"],
                num_rows="fixed",
                key="product_editor"  # Add a unique key
            )
            
            # Update session state with edited values and force rerun if changed
            new_fare = np.array(edited_product_data['Fare ($)'])
            new_demand = np.array(edited_product_data['Expected Demand'])
            
            if not np.array_equal(new_fare, st.session_state.fare) or not np.array_equal(new_demand, st.session_state.demand):
                st.session_state.fare = new_fare
                st.session_state.demand = new_demand
                st.rerun()  # Replace st.experimental_rerun() with st.rerun()
            
        with tab2:
            # Create editable leg data
            leg_data = pd.DataFrame({
                'Leg': [f'L{i+1}' for i in range(L)],
                'Capacity': st.session_state.capacity,
                'Cancellation Probability': st.session_state.cancel_prob
            })
            
            st.subheader("Leg Capacity and Cancellation Rates")
            edited_leg_data = st.data_editor(
                leg_data, 
                use_container_width=True,
                hide_index=True,
                disabled=["Leg"],
                num_rows="fixed",
                key="leg_editor"  # Add a unique key
            )
            
            # Update session state with edited values and force rerun if changed
            new_capacity = np.array(edited_leg_data['Capacity'])
            new_cancel_prob = np.array(edited_leg_data['Cancellation Probability'])
            
            if not np.array_equal(new_capacity, st.session_state.capacity) or not np.array_equal(new_cancel_prob, st.session_state.cancel_prob):
                st.session_state.capacity = new_capacity
                st.session_state.cancel_prob = new_cancel_prob
                st.rerun()  # Replace st.experimental_rerun() with st.rerun()
            
        with tab3:
            st.subheader("Product-to-Legs Mapping")
            
            # Create a more human-readable representation of the mapping
            readable_mapping = []
            for p in range(P):
                legs_used = leg_finder(p+1, product_to_legs)
                readable_mapping.append({
                    'Product': f'P{p+1}',
                    'Legs Used': ', '.join([f'L{leg}' for leg in legs_used])
                })
            
            st.dataframe(pd.DataFrame(readable_mapping), use_container_width=True)
            
            # Advanced option to edit the raw product_to_legs matrix
            st.subheader("Advanced: Edit Raw Product-to-Legs Matrix")
            if st.checkbox("Show advanced network editor"):
                st.warning("""
                Use caution when editing this matrix. Each row represents a product.
                Positive numbers represent legs used by the product, and -1 means no leg.
                Example: [1, -1] means the product uses only leg 1, while [2, 3] means it uses legs 2 and 3.
                """)
                
                # Convert to string for easier editing
                ptl_strings = [str(row) for row in product_to_legs]
                ptl_df = pd.DataFrame({
                    'Product': [f'P{i+1}' for i in range(P)],
                    'Legs Matrix': ptl_strings
                })
                
                edited_ptl_df = st.data_editor(
                    ptl_df,
                    use_container_width=True,
                    hide_index=True,
                    disabled=["Product"],
                    num_rows="fixed"
                )
                
                # Parse the edited strings back to numpy arrays
                try:
                    edited_product_to_legs = np.array([
                        eval(row_str) for row_str in edited_ptl_df['Legs Matrix']
                    ])
                    # Check if the shape is valid
                    if edited_product_to_legs.shape == product_to_legs.shape:
                        st.session_state.product_to_legs = edited_product_to_legs
                    else:
                        st.error("The shape of the product-to-legs matrix has changed. Please maintain the same format.")
                except Exception as e:
                    st.error(f"Error parsing the product-to-legs matrix: {str(e)}. Please check your input format.")

    # --- Main App with Run Button ---
    st.button("Run Optimization", type="primary", on_click=run_optimization)

    # Show results when optimization has been run
    if st.session_state.optimization_run:
        # Get current data from session state
        fare = st.session_state.fare
        demand = st.session_state.demand
        capacity = st.session_state.capacity
        cancel_prob = st.session_state.cancel_prob
        product_to_legs = st.session_state.product_to_legs
        
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

        # Add visualization button with session state
        st.button("Show Visualizations", on_click=show_visualizations)

        # Show visualizations if the button has been clicked
        if st.session_state.show_viz:
            st.header("Data Visualizations")

            # DAVN Matrix Heatmap
            st.subheader("DAVN Bid-Price Matrix Visualization")
            davn_fig = plot_davn_heatmap(df_davn.values)
            st.pyplot(davn_fig)

            # Booking Limits Bar Charts
            st.subheader("Booking Limits by Leg")
            bl_fig = plot_booking_limits(booking_table)
            st.pyplot(bl_fig)
    else:
        st.info("Click the 'Run Optimization' button to calculate DAVN bid prices and booking limits.")
