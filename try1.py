from exdata import *
import numpy as np
import dsd
import networkx as nx
from dsd import *
from datetime import datetime
import matplotlib.pyplot as plt
from greedypeeling import charikar_peeling
from scipy.sparse import coo_matrix, tril, csr_matrix

epsilon = 3.0

def randomized_response(value, epsilon):
    """随机响应机制：对单个二值元素进行随机响应处理"""
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    noise = np.random.rand() < p  # 以 p 的概率保留原值，以 1-p 的概率反转
    return value if noise else 1 - value

def add_dp_noise_to_sparse_matrix(sparse_matrix, epsilon):
    """对稀疏矩阵的下三角部分应用随机响应保护"""
    # 确保输入矩阵为 float 类型，避免 object 类型错误
    sparse_matrix = sparse_matrix.astype(float)

    # 获取下三角部分的稀疏矩阵
    lower_tri = tril(sparse_matrix, k=-1).tocoo()  # k=-1 表示严格下三角

    # 对下三角部分的非零位置应用随机响应机制
    noisy_data = [randomized_response(v, epsilon) for v in lower_tri.data]

    # 用带噪声的数据构建新的下三角稀疏矩阵
    noisy_lower_tri = coo_matrix((noisy_data, (lower_tri.row, lower_tri.col)), shape=sparse_matrix.shape)

    # 生成对称的稀疏矩阵：加上它的转置形成对称矩阵
    symmetric_noisy_matrix = noisy_lower_tri + noisy_lower_tri.transpose()

    return symmetric_noisy_matrix

# 创建无向图对象
G = nx.Graph()

# 读取文件并添加边
with open("/Users/teco/Pych/DSP/DSP/datasets/IMDB/edges.csv", "r") as f:
    for line in f:
        # 去掉行尾换行符并按逗号分割每一行
        node1, node2 = map(int, line.strip().split(","))
        # 添加边到图中
        G.add_edge(node1, node2)

# 将图转换为稀疏矩阵
sparse_adj_matrix = nx.to_scipy_sparse_array(G, format='csr')

# 第一次应用差分隐私保护
dp_protected_matrix = add_dp_noise_to_sparse_matrix(sparse_adj_matrix, epsilon)

# 第二次应用差分隐私保护
ldp_protected_matrix = add_dp_noise_to_sparse_matrix(dp_protected_matrix, epsilon)

# 将生成的稀疏矩阵转换为 NetworkX 图
ldp_noisy_graph = nx.from_scipy_sparse_array(ldp_protected_matrix)

# 原始图的 dense subgraph 计算
print("greedypeeling method")
start = datetime.now()
dense_subgraph, density = charikar_peeling(G)
print('Run time:', datetime.now() - start, '\n')
print("Nodes in dense subgraph:", dense_subgraph.nodes())
print("Density of the dense subgraph:", density)

# 差分隐私保护后的噪声图 dense subgraph 计算
print("greedypeeling method on noisy graph")
start = datetime.now()
dense_subgraph_N, density_N = charikar_peeling(ldp_noisy_graph)
print('Run time:', datetime.now() - start, '\n')
print("Nodes in dense subgraph (noisy):", dense_subgraph_N.nodes())
print("Density of the dense subgraph in LDP:", density_N)

# 打印结果
print("Density of the dense subgraph (original):", density)
print("Density of the dense subgraph in LDP:", density_N)