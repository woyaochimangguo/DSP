import numpy as np
import networkx as nx
def randomized_response(value, epsilon):
    """随机响应机制：对单个二值元素进行随机响应处理"""
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    noise = np.random.rand() < p  # 以 p 的概率保留原值，以 1-p 的概率反转
    return value if noise else 1 - value

def add_dp_noise_to_sparse_matrix(sparse_matrix, epsilon):
    """对稀疏矩阵的下三角部分应用随机响应保护"""
    # 获取下三角部分的稀疏矩阵
    lower_tri = tril(sparse_matrix, k=-1).tocoo()  # k=-1 表示严格下三角

    # 对每个下三角的非零位置应用随机响应机制
    noisy_data = [randomized_response(v, epsilon) for v in lower_tri.data]

    # 用带噪声的数据构建新的下三角稀疏矩阵
    noisy_lower_tri = coo_matrix((noisy_data, (lower_tri.row, lower_tri.col)), shape=sparse_matrix.shape)

    # 生成对称的稀疏矩阵：加上它的转置形成对称矩阵
    symmetric_noisy_matrix = noisy_lower_tri + noisy_lower_tri.transpose()

    return symmetric_noisy_matrix
from scipy.sparse import coo_matrix, tril, csr_matrix

# 生成稀疏邻接矩阵
# sparse_adj_matrix = generate_sparse_adjacency_matrix(G)
def charikar_peeling(G):
    """
    Charikar's peeling algorithm to find a dense subgraph.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    tuple: A tuple containing the dense subgraph and its density.
    """
    # Copy the original graph to avoid modifying it
    G = G.copy()

    # Initialize variables to track the best density and corresponding subgraph
    best_density = 0
    best_subgraph = None

    # Compute initial density
    def compute_density(G):
        num_edges = G.number_of_edges()
        num_nodes = G.number_of_nodes()
        return num_edges / num_nodes if num_nodes > 0 else 0



    # Iteratively remove the node with the smallest degree
    while G.number_of_nodes() > 0:
        current_density = compute_density(G)
        print("现在的密度为：",current_density)
        if current_density > best_density:
            best_density = current_density
            best_subgraph = G.copy()
            print("current density is", current_density)
            print("current subgraph nodes is ",best_subgraph)
        # Find the node with the smallest degree
        min_degree_node = min(G.nodes, key=G.degree)
        G.remove_node(min_degree_node)

    return best_subgraph, best_density


def charikar_peeling_RR(G):
    """
    Charikar's peeling algorithm to find a dense subgraph.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    tuple: A tuple containing the dense subgraph and its density.
    """
    # Copy the original graph to avoid modifying it
    G = G.copy()

    # Initialize variables to track the best density and corresponding subgraph
    best_density = 0
    best_subgraph = None

    # Compute initial density
    def compute_density(G):
        epsilon = 4
        p = 1 / (np.exp(epsilon) + 1)
        no_flip_prob = 1 - p  # 不翻转的概率
        num_edges = (1-p)*len(G.edges) + p * (0.5*len(G.nodes)*(len(G.nodes)-1) - len(G.edges))
        num_nodes = G.number_of_nodes()
        return num_edges / num_nodes if num_nodes > 0 else 0

    # Iteratively remove the node with the smallest degree
    while G.number_of_nodes() > 0:
        epsilon = 4
        p = 1 / (np.exp(epsilon) + 1)
        current_density = compute_density(G)
        print("现在的密度为：",current_density)
        if current_density > best_density:
            best_density = current_density
            best_subgraph = G.copy()
            print("current density is", current_density)
            print("current subgraph nodes is ",best_subgraph)
        # Find the node with the smallest degree
        expected_degrees = {}
        for node in G.nodes:
            original_degree = G.degree[node]
            num_non_neighbors = len(G.nodes) - 1 - original_degree
            expected_degree = original_degree * (1-p) + num_non_neighbors * p
            expected_degrees[node] = expected_degree
        min_degree_node = min(G.nodes, key=lambda node: expected_degrees[node])
        print("该最小度数节点的原始噪声度数是:",G.degree[min_degree_node])
        print("该最小度数节点的期望度数是：",expected_degrees[min_degree_node])
        G.remove_node(min_degree_node)

    return best_subgraph, best_density

def generate_adjacency_matrix(graph):
    """
    Generate the adjacency matrix of a given graph.

    Parameters:
    graph (networkx.Graph): The input graph.

    Returns:
    np.ndarray: Adjacency matrix of the graph.
    """
    return nx.to_numpy_array(graph, dtype=int)


def add_random_response_noise(matrix, epsilon):
    """
    Apply Random Response (RR) algorithm to the lower triangular part of the adjacency matrix.

    Parameters:
    matrix (np.ndarray): Original adjacency matrix.
    epsilon (float): Privacy budget parameter.

    Returns:
    np.ndarray: Noisy adjacency matrix.
    """
    n = matrix.shape[0]
    noisy_matrix = matrix.copy()
    p = np.exp(epsilon) / (1 + np.exp(epsilon))  # Random response probability

    for i in range(n):
        for j in range(i):  # Iterate over lower triangular part
            if np.random.rand() < p:
                noisy_matrix[i, j] = matrix[i, j]
            else:
                noisy_matrix[i, j] = 1 - matrix[i, j]
            noisy_matrix[j, i] = noisy_matrix[i, j]  # Ensure symmetry

    return noisy_matrix


# Main code
# 创建无向图对象
G = nx.Graph()
# 读取文件并添加边
with open("C:\\Users\\ECNU\\Documents\\GitHub\\DSP\\datasets\\IMDB\\edges.csv", "r") as f:
    for line in f:
        # 去掉行尾换行符并按逗号分割每一行
        node1, node2 = map(int, line.strip().split(","))
        # 添加边到图中
        G.add_edge(node1, node2)
#采用稀疏矩阵的方式存储图
if hasattr(nx, 'to_scipy_sparse_array'):
    # 使用新的函数来转换图为稀疏矩阵
    adj_matrix = nx.to_scipy_sparse_array(G, format='coo')
else:
    # 使用旧的函数（可能是旧版本的 NetworkX）
    adj_matrix = nx.to_scipy_sparse_matrix(G, format='coo')

# # 图构建完成
# print("Graph has been created.")
# print("Number of nodes:", G.number_of_nodes())
# print("Number of edges:", G.number_of_edges())
# G = nx.read_edgelist('./datasets/Facebook/facebook/107.edges', nodetype=int)
print("原始图的节点数为：",len(G.nodes))
print("原始图的边数为：",len(G.edges))
# Generate adjacency matrix(G)
adj_matrix = generate_sparse_adjacency_matrix(G)
print("Original Adjacency Matrix:\n", adj_matrix)

# Apply Random Response for differential privacy
epsilon = 4.0  # Privacy budget
p = 1 / (1 + np.exp(epsilon))
noisy_adj_matrix = add_random_response_noise(adj_matrix, epsilon)
print("Noisy Adjacency Matrix:\n", noisy_adj_matrix)
G_N = nx.from_numpy_matrix(noisy_adj_matrix)
noisy_adj_matrix_N = add_random_response_noise(noisy_adj_matrix, epsilon)
G_N_N = nx.from_numpy_matrix(noisy_adj_matrix_N)
print("噪声图的节点数：",len(G_N.nodes))
print("噪声图的边数：",len(G_N.edges))
print("噪声图纠偏以后的节点数：",len(G_N.nodes))
print("噪声图纠偏以后的边数：",len(G_N.edges))
# print("噪声图中存在的边的期望为",(1-p)*len(G_N.edges))
# print("噪声图中不存在的边的数量为",0.5*len(G_N.nodes)*(len(G_N.nodes)-1) - len(G_N.edges))
# print("噪声图中不存在的边原始图存在的期望为：",p * (0.5*len(G_N.nodes)*(len(G_N.nodes)-1) - len(G_N.edges)))
# print("噪声图的期望边数：",(1-p)*len(G_N.edges) + p * (0.5*len(G_N.nodes)*(len(G_N.nodes)-1) - len(G_N.edges)) )
dense_subgraph,density = charikar_peeling(G)
dense_subgraph_N,density_N = charikar_peeling(G_N)
dense_subgraph_N_N,density_N_N = charikar_peeling(G_N_N)
# dense_subgraph_RR,density_RR = charikar_peeling_RR(G_N)
print("原始图的最密子图密度：",density)
print("原始图的最密子集的集合为：",dense_subgraph.nodes)
print("噪声图的最密子图密度：",density_N)
print("噪声图的最密子图的集合为：",dense_subgraph_N.nodes)
print("噪声图的最密子图密度：",density_N_N)
print("噪声图的最密子图的集合为：",dense_subgraph_N_N.nodes)
# print("噪声图最密子图的期望密度",density_RR)
# print("噪声图最密子图的期望集合",dense_subgraph_RR.nodes)
# print("噪声图的密度为：",G_N.number_of_edges() / G_N.number_of_nodes())