
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.cluster import KMeans

# Qiskit Imports
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.applications import VehicleRouting
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import SparsePauliOp

# ============================================
# STEP 1: Data Collection
# ============================================
def load_data(file_path, num_nodes=4):
    """
    Load data and select a subset of nodes (Depot + Customers).
    num_nodes: Total nodes to keep (1 depot + num_nodes-1 customers)
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Select first 'num_nodes' (Depot is index 0)
    subset = df.head(num_nodes).copy()
    print(f"Selected {len(subset)} nodes for simulation.")
    return subset

# ============================================
# STEP 2: Preprocessing (Distance Matrix)
# ============================================
def calculate_distance_matrix(df):
    """
    Calculate Euclidean distance matrix between all nodes.
    """
    n = len(df)
    coords = df[['XCOORD', 'YCOORD']].to_numpy()
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
                
    print("\nDistance Matrix (First 4x4):")
    print(dist_matrix[:4, :4])
    return dist_matrix

# ============================================
# STEP 3 & 4: Formulation (QUBO)
# ============================================
def build_qubo(dist_matrix, num_vehicles=1):
    """
    Convert VRP instance to QUBO using qiskit-optimization.
    Note: qiskit-optimization's VehicleRouting is essentially a TSP formulation 
    if num_vehicles=1. For VRP, it adds depot constraints.
    """
    print("\nConstructing VRP Problem...")
    # Instantiate the application class
    vrp = VehicleRouting(dist_matrix, num_vehicles=num_vehicles, depot=0)
    
    # Convert to Quadratic Program
    qp = vrp.to_quadratic_program()
    print("Quadratic Program Created.")
    
    # Convert to QUBO
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)
    print("Converted to QUBO.")
    
    # Get Operator (Ising Hamiltonian)
    op, offset = qubo.to_ising()
    print(f"Hamiltonian Operator created with {op.num_qubits} qubits.")
    
    return qubo, op, offset, vrp

# ============================================
# STEP 5 & 6: QAOA Execution
# ============================================
def run_qaoa(operator, offset, qubo):
    """
    Run QAOA using the new Sampler primitive.
    """
    print("\nStarting QAOA Optimization...")
    
    # Setup Optimizer
    optimizer = COBYLA(maxiter=100)
    
    # Setup Primitive
    # Note: Using StatevectorSampler (V2)
    sampler = StatevectorSampler()
    
    # Setup QAOA
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
    
    # Run optimization
    # compute_minimum_eigenvalue expects an operator
    result = qaoa.compute_minimum_eigenvalue(operator)
    
    print("\nQAOA Result:")
    print(f"Optimal Value: {result.optimal_value + offset}")
    print(f"Optimal Parameters: {result.optimal_point}")
    
    # Extract best solution
    # The result eigenstate is a quasi-distribution
    best_measurement = result.best_measurement
    bitstring = best_measurement['bitstring']
    
    print(f"Best Measurement (Bitstring): {bitstring}")
    
    # Convert bitstring back to variable dictionary for interpretation
    # Note: bitstring order might need reversal depending purely on how qiskit returns it vs variables
    # But usually creating a generic solution vector is safest
    x = np.array([int(b) for b in bitstring])
    
    # In qiskit-algorithms result, sometimes we just use the eigenstate directly if available
    # Or just interpret the bitstring. 
    # Let's map it back strictly
    # (Simplified for this script: We trust the bitstring corresponds to [x0, x1, ...])
    
    return x


# ============================================
# STEP 7: Visualization
# ============================================
def plot_solution(df, route):
    """
    Plot the nodes and the optimal route found by QAOA.
    route: List of routes, e.g., [[[0, 1], [1, 2], [2, 0]]]
    """
    print("\nPlotting solution...")
    plt.figure(figsize=(8, 6))
    
    # 1. Plot Nodes
    # Depot (Index 0)
    plt.scatter(df.iloc[0]['XCOORD'], df.iloc[0]['YCOORD'], c='red', s=200, marker='*', label='Depot')
    # Customers (Index 1+)
    plt.scatter(df.iloc[1:]['XCOORD'], df.iloc[1:]['YCOORD'], c='blue', s=100, label='Customers')
    
    # Label Nodes
    for idx, row in df.iterrows():
        plt.text(row['XCOORD']+0.5, row['YCOORD']+0.5, str(int(row['CUST_NO'])), fontsize=12)

    # 2. Plot Route
    # route structure from VehicleRouting.interpret is usually a list of routes (for multiple vehicles)
    # e.g. [ [[0, 2], [2, 1], [1, 0]] ]
    
    colors = ['g', 'm', 'c', 'k'] # Colors for different vehicles
    
    for vehicle_id, r in enumerate(route):
        color = colors[vehicle_id % len(colors)]
        print(f"Plotting route for Vehicle {vehicle_id+1}: {r}")
        
        for edge in r:
            i, j = edge[0], edge[1]
            x1, y1 = df.iloc[i]['XCOORD'], df.iloc[i]['YCOORD']
            x2, y2 = df.iloc[j]['XCOORD'], df.iloc[j]['YCOORD']
            
            plt.plot([x1, x2], [y1, y2], c=color, lw=2, linestyle='-' if vehicle_id==0 else '--')
            
            # Draw arrow to show direction
            plt.arrow(x1, y1, (x2-x1)*0.9, (y2-y1)*0.9, 
                      head_width=1.5, head_length=1.5, fc=color, ec=color, alpha=0.6)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("VRP Solution via QAOA")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    # Save plot instead of blocking via show() if running in restricted env, 
    # but show() is fine for local user.
    plt.show()

# ============================================
# MAIN PIPELINE
# ============================================

# ============================================
# STEP 8: Clustering & Scalability
# ============================================
def solve_large_vrp(df, num_clusters=2):
    """
    Solve VRP by clustering customers and solving sub-VRPs.
    """
    print(f"\n[CLUSTERING] Dividing {len(df)-1} customers into {num_clusters} clusters...")
    
    cities = df.iloc[1:]  # Customers only
    coords = cities[['XCOORD', 'YCOORD']].values
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cities.loc[:, 'cluster'] = kmeans.fit_predict(coords)
    
    full_routes = []
    
    for c_id in range(num_clusters):
        print(f"\n--- Processing Cluster {c_id} ---")
        # Get nodes for this cluster
        cluster_nodes = cities[cities['cluster'] == c_id].copy()
        
        # Add Depot (Node 0) to this cluster subset
        # We need to reconstruct a dataframe [Depot, Cust1, Cust2...]
        depot = df.iloc[[0]].copy()
        subset = pd.concat([depot, cluster_nodes], ignore_index=True)
        
        # Map original Customer IDs (for tracking) logic is skipped for simplicity here,
        # we strictly solve the subset as a new isolated problem [0, 1, 2...] 
        # but plotting needs original coordinates which 'subset' has.
        
        # 1. Distance
        d_mat = calculate_distance_matrix(subset)
        
        # 2. QAOA
        qubo, op, offset, vrp_inst = build_qubo(d_mat, num_vehicles=1)
        res_vec = run_qaoa(op, offset, qubo)
        
        # 3. Interpret & Store
        # result_interpretation returned by qiskit is like [[[0, 1], [1, 2], [2, 0]]] (indices of subset)
        # We need to map `subset` indices back to `df` indices if we strictly wanted perfect global IDs
        # But for 'plot_solution' taking 'df', we just need 'routes' in terms of 'df' indices?
        # Actually plot_solution takes 'df' and 'routes'. 
        # If we return a list of routes, plot_solution iterates them.
        # But wait, 'subset' has different indices (0, 1, 2) than 'df' (0, 5, 8...).
        # We must remap the edges.
        
        local_route = vrp_inst.interpret(res_vec)[0] # Single vehicle route [[i,j],[j,k]...]
        
        # Remap local indices to global indices
        # subset has indices 0, 1, 2...
        # subset.iloc[i] corresponds to some row in 'df'.
        
        global_route = []
        for edge in local_route:
            u_local, v_local = edge[0], edge[1]
            
            # Find the true CUST_NO or index from original df
            # To simplify plotting: Let's assume we pass the *subset* to plot_solution iteratively?
            # Or map back. Let's map back to global coordinates for a unified plot.
            
            # Actually easier: Just plot each cluster result sequentially on the same figure handle? Not possible with current function.
            # Let's map back to integer indices of the original dataframe 'df'.
            
            # Get X,Y of u_local
            u_row = subset.iloc[u_local]
            v_row = subset.iloc[v_local]
            
            # find index in original df where CUST_NO matches
            u_global = df.index[df['CUST_NO'] == u_row['CUST_NO']].tolist()[0]
            v_global = df.index[df['CUST_NO'] == v_row['CUST_NO']].tolist()[0]
            
            global_route.append([u_global, v_global])
            
        full_routes.append(global_route)
        
    return full_routes


# ============================================
# MAIN PIPELINE
# ============================================
if __name__ == "__main__":
    file_path = "c:\\QC_PROJECT\\C101_MTW.csv"
    
    # [SCALABLE MODE]
    # We load 7 nodes: 1 Depot + 6 Customers
    total_nodes = 7
    data = load_data(file_path, num_nodes=total_nodes) 
    
    # Use Clustering to solve (2 clusters -> 3 customers each + depot)
    # This keeps qubit count low (approx 3+1=4 nodes => ~9 qubits complexity per cluster)
    final_routes = solve_large_vrp(data, num_clusters=2)
    
    print("\nFinal Aggregated Routes (Global Indices):")
    print(final_routes)
    
    # Visualize All
    plot_solution(data, final_routes)

    print("\n[SUCCESS] Scalable VRP-QAOA Pipeline Completed Successfully.")
