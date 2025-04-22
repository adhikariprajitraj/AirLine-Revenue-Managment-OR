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
    prot = np.zeros(n)
    for i in range(1, n):
        if agg_f[i] > fares[i - 1]:
            frac = (agg_f[i] - fares[i-1]) / agg_f[i]
            prot[i] = poisson.ppf(frac, agg_d[i])
    # prot[0] unused

    # 3) Adjust capacity for cancellations
    C_eff = capacity / (1 - cancel_prob)

    # 4) Booking limits: shift protection by one
    bl = np.zeros(n)
    # For classes 1…n−1
    bl[:-1] = np.maximum(C_eff - prot[1:], 0)
    # Highest fare has full capacity
    bl[-1] = C_eff

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
            davn[p-1, l-1] = fare[p-1] - psi + shadow_prices[l-1]
    return davn

def plot_davn_heatmap(davn_data):
    # Create a copy to avoid modifying the original
    davn_copy = davn_data.copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a mask for the -1 values
    mask = davn_copy == -1
    
    # Create heatmap with custom colormap and improved annotations
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
    
    # Add row and column annotations for better understanding
    ax.set_title('DAVN Bid-Price Matrix\n(Virtual fare value of each product on each leg)', fontsize=16)
    ax.set_xlabel('Flight Legs', fontsize=12)
    ax.set_ylabel('Products', fontsize=12)
    
    # Add explanation text
    fig.text(0.5, 0.01, 
             "Higher values (yellow) indicate more valuable products for a specific leg.\n"
             "Empty cells (-1) indicate product doesn't use that leg.",
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for caption
    
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
        
        # Create color map based on virtual fare values
        virtual_fares = leg_data['Virtual Fare'].values
        if len(virtual_fares) > 0:
            norm = plt.Normalize(min(virtual_fares), max(virtual_fares))
            colors = plt.cm.coolwarm(norm(virtual_fares))
        else:
            colors = 'skyblue'
        
        # Create the bar chart with gradient colors
        bars = axs[i].bar(
            ["P" + str(p) for p in leg_data['Product']],
            leg_data['Booking Limit'],
            color=colors
        )
        
        # Add virtual fare labels above bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            # Add virtual fare above the bar
            axs[i].text(
                bar.get_x() + bar.get_width()/2.,
                height + 5,
                f"${leg_data['Virtual Fare'].iloc[j]}",
                ha='center', va='bottom',
                rotation=0,
                fontsize=9
            )
            
            # Add booking limit inside the bar
            if height > 0:  # Only add text if the bar is tall enough
                axs[i].text(
                    bar.get_x() + bar.get_width()/2.,
                    height/2,  # Middle of the bar
                    f"{int(leg_data['Booking Limit'].iloc[j])}",
                    ha='center', va='center',
                    color='white' if virtual_fares[j] > np.mean(virtual_fares) else 'black',
                    fontweight='bold',
                    fontsize=10
                )
        
        axs[i].set_title(f'Leg {leg} Booking Limits', fontsize=14)
        axs[i].set_xlabel('Products')
        axs[i].set_ylabel('Booking Limit')
        axs[i].grid(axis='y', alpha=0.3)
        
        # Add capacity line
        capacity_value = capacity[leg-1]
        axs[i].axhline(y=capacity_value, color='r', linestyle='-', alpha=0.7)
        axs[i].text(0, capacity_value+2, f"Capacity: {capacity_value}", color='r')
        
        # Add note about color coding
        if i == 0:
            axs[i].text(
                0.5, -0.15,
                "Color indicates virtual fare value: Blue (lower) to Red (higher)",
                transform=axs[i].transAxes,
                ha='center', fontsize=9, style='italic'
            )
    
    plt.tight_layout()
    return fig

def plot_revenue_distribution(x_opt, fare, obj):
    """Create a revenue distribution visualization"""
    
    # Calculate revenue by product
    revenue_by_product = x_opt * fare
    
    # Sort products by revenue contribution
    sorted_indices = np.argsort(revenue_by_product)[::-1]  # Descending
    top_products = sorted_indices[:15]  # Show top 15 products
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create pie chart of revenue distribution
    pie_labels = [f'P{i+1}' for i in top_products]
    pie_values = revenue_by_product[top_products]
    other_value = sum(revenue_by_product) - sum(pie_values)
    
    if other_value > 0:
        pie_labels.append('Others')
        pie_values = np.append(pie_values, other_value)
    
    # Use a nice color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(pie_values)))
    
    # Create the pie chart
    wedges, texts, autotexts = ax1.pie(
        pie_values, 
        labels=pie_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    
    # Make percentage labels more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax1.set_title('Revenue Distribution by Product', fontsize=14)
    
    # Create horizontal bar chart for top revenue products
    bars = ax2.barh(
        [f'P{i+1}' for i in top_products[:10]],  # Top 10 for readability
        revenue_by_product[top_products[:10]],
        color=colors[:10]
    )
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax2.text(
            width + 500,  # Small offset
            bar.get_y() + bar.get_height()/2,
            f'${int(width):,}',
            ha='left', va='center'
        )
    
    ax2.set_xlabel('Revenue ($)')
    ax2.set_title('Top 10 Products by Revenue Contribution', fontsize=14)
    
    # Add total revenue annotation
    fig.text(0.5, 0.01, f'Total Expected Revenue: ${int(obj):,}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig

def plot_demand_vs_capacity(demand, capacity, product_to_legs, fare):
    """Visualize demand vs. capacity and fare distribution"""
    
    # Calculate total demand per leg
    leg_demand = {}
    for l in range(1, len(capacity)+1):
        products_on_leg = [j for j in range(len(demand)) if l in leg_finder(j+1, product_to_legs)]
        leg_demand[l] = sum(demand[j] for j in products_on_leg)
    
    # Set up figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Demand vs Capacity Chart
    leg_nums = list(leg_demand.keys())
    demand_values = [leg_demand[l] for l in leg_nums]
    capacity_values = [capacity[l-1] for l in leg_nums]
    
    # Create bar positions
    x = np.arange(len(leg_nums))
    width = 0.35
    
    # Plot bars
    demand_bars = ax1.bar(x - width/2, demand_values, width, label='Expected Demand', color='skyblue')
    capacity_bars = ax1.bar(x + width/2, capacity_values, width, label='Capacity', color='lightcoral')
    
    # Add value labels on bars
    for bar in demand_bars + capacity_bars:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Add capacity utilization percentage
    for i, l in enumerate(leg_nums):
        utilization = (leg_demand[l] / capacity[l-1]) * 100
        ax1.text(i, 0, f"{utilization:.1f}%", ha='center', va='bottom', 
                fontsize=9, fontweight='bold', color='darkblue',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # Customize plot
    ax1.set_title('Demand vs. Capacity by Leg', fontsize=14)
    ax1.set_xlabel('Leg')
    ax1.set_ylabel('Number of Seats')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Leg {l}' for l in leg_nums])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Fare Distribution Analysis
    # Calculate statistics for fare distribution by leg
    fare_stats = {}
    for l in range(1, len(capacity)+1):
        products_on_leg = [j for j in range(len(fare)) if l in leg_finder(j+1, product_to_legs)]
        if products_on_leg:
            fares_on_leg = fare[products_on_leg]
            fare_stats[l] = {
                'min': np.min(fares_on_leg),
                'max': np.max(fares_on_leg),
                'avg': np.mean(fares_on_leg),
                'median': np.median(fares_on_leg)
            }
    
    # Create box plot for fare distribution
    boxplot_data = [fare[
        [j for j in range(len(fare)) if l in leg_finder(j+1, product_to_legs)]
    ] for l in leg_nums]
    
    ax2.boxplot(boxplot_data, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen', alpha=0.8),
                medianprops=dict(color='darkred'),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5))
    
    # Customize plot
    ax2.set_title('Fare Distribution by Leg', fontsize=14)
    ax2.set_xlabel('Leg')
    ax2.set_ylabel('Fare Value ($)')
    ax2.set_xticklabels([f'Leg {l}' for l in leg_nums])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add annotations for key statistics
    for i, l in enumerate(leg_nums):
        if l in fare_stats:
            stats = fare_stats[l]
            y_pos = stats['min'] - 50
            ax2.annotate(f"Avg: ${stats['avg']:.0f}\nRange: ${stats['max'] - stats['min']:.0f}",
                        xy=(i+1, y_pos),
                        ha='center', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # Add overall insights
    average_util = sum(leg_demand[l] / capacity[l-1] for l in leg_nums) / len(leg_nums) * 100
    fig.suptitle(f'Capacity Utilization Analysis (Avg: {average_util:.1f}%)', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

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

if __name__ == "__main__":
    # Set up session state to track app state
    if "optimization_run" not in st.session_state:
        st.session_state.optimization_run = False
    if "show_viz" not in st.session_state:
        st.session_state.show_viz = False
    if "show_input" not in st.session_state:
        st.session_state.show_input = False

    def run_optimization():
        st.session_state.optimization_run = True
        st.session_state.show_viz = False

    def show_visualizations():
        st.session_state.show_viz = True
        
    def toggle_input_data():
        st.session_state.show_input = not st.session_state.show_input

    st.set_page_config(layout="wide")
    st.title("DAVN & EMSR-b Airline Revenue Management")

    # --- Description ---
    st.markdown("""
    ## Optimizing Seat Allocation to Maximize Airline Revenue

    Airlines face a daily challenge: how to best manage limited seat capacity on their flights while maximizing revenue. The key lies in offering the right seats, at the right price, to the right customers — and knowing when to say no to low-fare bookings in hopes of selling those seats at a higher price later.

    This is the essence of airline revenue management. Airlines don't just sell one type of ticket — they offer multiple fare classes for the same route, ranging from economy saver fares to flexible business class tickets. Each fare class has different prices and conditions, and customer demand for these fares fluctuates over time.

    To make the most out of every flight, airlines use smart optimization techniques to decide:
    - How many seats should be available at each fare level?
    - When should a cheaper fare stop being offered?
    - How can the value of a seat be assessed when multiple itineraries overlap across  network of flights?

    The EMSR-b rule helps airlines set limits on how many tickets to sell at each price level. It uses historical demand data to strike a balance: selling seats early at a lower price versus waiting to sell them later at a higher price.

    However, in real-world airline operations, things are more complicated. Many customers book trips that involve multiple connecting flights, and a seat on one leg of a journey may be shared across many different itineraries.

    To handle this, our approach uses a method called Displacement Adjusted Virtual Nesting (DAVN). This method helps airlines make smarter decisions by considering the entire flight network. It estimates the true revenue value of each ticket, accounting for the opportunity cost of assigning a seat to one itinerary instead of another.

    Using a combination of:
    - Linear programming to optimize revenue across the network, and
    - Seat-allocation principles based on pricing and demand to manage booking decisions at the individual flight level,

    This integrated system enables airlines to manage bookings in a way that is both strategically optimal and operationally practical.
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

    # Add button to show/hide input data
    st.button("Show/Hide Input Data", on_click=toggle_input_data)
    
    # Display input data if toggled
    if st.session_state.show_input:
        st.header("Input Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Product Fares ($)")
            df_fare = pd.DataFrame({
                'Product': [f'P{i+1}' for i in range(P)],
                'Fare ($)': fare
            })
            st.dataframe(df_fare, use_container_width=True)
            
            st.subheader("Leg Capacity")
            df_capacity = pd.DataFrame({
                'Leg': [f'L{i+1}' for i in range(L)],
                'Capacity': capacity
            })
            st.dataframe(df_capacity, use_container_width=True)
        
        with col2:
            st.subheader("Product Demand (expected bookings)")
            df_demand = pd.DataFrame({
                'Product': [f'P{i+1}' for i in range(P)],
                'Demand': demand
            })
            st.dataframe(df_demand, use_container_width=True)
            
            st.subheader("Leg Cancellation Probability")
            df_cancel = pd.DataFrame({
                'Leg': [f'L{i+1}' for i in range(L)],
                'Cancellation Probability': cancel_prob
            })
            st.dataframe(df_cancel, use_container_width=True)
        
        st.subheader("Product-to-Legs Mapping")
        df_mapping = pd.DataFrame({
            'Product': [f'P{p+1}' for p in range(P)],
            'Legs Used': [', '.join([f'L{leg}' for leg in leg_finder(p+1, product_to_legs)]) for p in range(P)]
        })
        st.dataframe(df_mapping, use_container_width=True)

    # --- Main App with Run Button ---
    st.button("Run Optimization", type="primary", on_click=run_optimization)

    # Show results when optimization has been run
    if st.session_state.optimization_run:
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
            
            # Create tabs for different visualizations
            viz_tabs = st.tabs([
                "Booking Limits", 
                "DAVN Matrix", 
                "Revenue Distribution", 
                "Demand Analysis"
            ])
            
            with viz_tabs[0]:
                # Booking Limits Bar Charts
                st.subheader("Booking Limits by Leg")
                bl_fig = plot_booking_limits(booking_table)
                st.pyplot(bl_fig)
                
                st.markdown("""
                **Understanding this visualization:**
                - Each bar represents the booking limit for a product on this leg
                - The number inside the bar is the actual booking limit
                - The color of the bar indicates the virtual fare value (blue = lower, red = higher)
                - The red line indicates the physical capacity of the flight
                """)
            
            with viz_tabs[1]:
                # DAVN Matrix Heatmap
                st.subheader("DAVN Bid-Price Matrix Visualization")
                davn_fig = plot_davn_heatmap(df_davn.values)
                st.pyplot(davn_fig)
                
                st.markdown("""
                **Understanding this visualization:**
                - This heatmap shows the virtual fare value for each product (rows) on each leg (columns)
                - Higher values (yellow) indicate more valuable products for that leg
                - Empty cells (white) indicate the product doesn't use that leg
                - Virtual fares are calculated using displacement-adjusted bid prices
                """)
            
            with viz_tabs[2]:
                # Revenue Distribution Visualization
                st.subheader("Revenue Distribution Analysis")
                rev_fig = plot_revenue_distribution(x_opt, fare, obj)
                st.pyplot(rev_fig)
                
                st.markdown("""
                **Understanding this visualization:**
                - The pie chart shows the distribution of revenue across products
                - The bar chart shows the top products by revenue contribution
                - Products with higher revenue contribution are more critical to overall profitability
                """)
            
            with viz_tabs[3]:
                # Demand vs Capacity Analysis
                st.subheader("Demand and Capacity Analysis")
                demand_fig = plot_demand_vs_capacity(demand, capacity, product_to_legs, fare)
                st.pyplot(demand_fig)
                
                st.markdown("""
                **Understanding this visualization:**
                - Left chart: Compares expected demand against capacity for each leg
                  - Percentages show capacity utilization (demand/capacity)
                - Right chart: Shows fare distribution for products on each leg
                  - Box plots show min/max, median, and quartiles
                  - Annotations show average fare and fare range
                - High utilization legs (near or above 100%) are potential bottlenecks
                - Legs with wider fare ranges offer more opportunities for revenue optimization
                """)
    else:
        st.info("Click the 'Run Optimization' button to calculate DAVN bid prices and booking limits.")
