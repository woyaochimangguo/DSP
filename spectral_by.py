import numpy as np
import networkx as nx
from scipy.linalg import eigh
def compute_density(G, S):
    """
    Compute the density of the subgraph induced by the set S.

    Parameters:
    G (networkx.Graph): The input graph.
    S (set): A subset of nodes in the graph.

    Returns:
    float: The density of the subgraph induced by S.
    """
    subgraph = G.subgraph(S)
    num_edges = subgraph.number_of_edges()
    num_nodes = subgraph.number_of_nodes()
    return num_edges / num_nodes if num_nodes > 0 else 0

def general_sweep_algorithm_laplacian_final(G):
    """
    General Sweep Algorithm to find the densest subgraph using Laplacian matrix.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    tuple: A tuple containing the subset of nodes in the densest subgraph found and its density.
    """
    # Compute the Laplacian matrix
    laplacian_matrix = nx.laplacian_matrix(G).todense()
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(laplacian_matrix)
    print("未排序的特征值:")
    print(eigenvalues)
    print("\n未排序的特征向量:")
    print(eigenvectors)
    # Use the second smallest eigenvector (Fiedler vector)
    sorted_indices = np.argsort(eigenvalues)
    print("特征值大小的排序为：",sorted_indices)
    print("排序后的特征值列表：",eigenvalues[sorted_indices])
    v1 = eigenvectors[-3]
    print("当前特征值对应的特征向量:",eigenvectors[-3])
    # Sort nodes in nonincreasing (descending) order of v1(i)
    sorted_nodes = np.argsort(v1)
    nodes_sorted = []
    G_nodes=list(G.nodes)
    for i in sorted_nodes:
        nodes_sorted.append(G_nodes[i])
    print("sorted_node lists is",nodes_sorted)
    best_S = list()
    best_density = 0

    current_S = list()  # Initialize the current set
    node = list()
    for node in nodes_sorted:
        current_S.append(node)  # Add the node to the current set
        # print("现在的子集S", current_S)
        # print("生成的子图的边",list(G.subgraph(current_S).edges))
        current_density = compute_density(G, current_S)  # Compute the density
        #print("current density is",current_density)
        # Update best subset if current density is greater
        if current_density > best_density:
            best_S = current_S.copy()  # Update best set
            best_density = current_density  # Update best density
        print("现在子集的密度：", best_density)

    return best_S, best_density  # Return both the subset and its density