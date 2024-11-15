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
    # 获取下三角部分的稀疏矩阵
    lower_tri = tril(sparse_matrix, k=-1).tocoo()  # k=-1 表示严格下三角

# 创建无向图对象
G = nx.Graph()
# 读取文件并添加边
with open("/Users/teco/Pych/DSP/DSP/datasets/IMDB/edges.csv", "r") as f:
    for line in f:
        # 去掉行尾换行符并按逗号分割每一行
        node1, node2 = map(int, line.strip().split(","))
        # 添加边到图中
        G.add_edge(node1, node2)
#采用稀疏矩阵的方式存储图
# if hasattr(nx, 'to_scipy_sparse_array'):
#     # 使用新的函数来转换图为稀疏矩阵
#     adj_matrix = nx.to_scipy_sparse_array(G, format='coo')
# else:
#     # 使用旧的函数（可能是旧版本的 NetworkX）
#     adj_matrix = nx.to_scipy_sparse_matrix(G, format='coo')

sparse_adj_matrix = nx.to_scipy_sparse_array(G, format='csr')
#进行第一次翻转
dp_protected_matrix = add_dp_noise_to_sparse_matrix(sparse_adj_matrix, epsilon)
#进行第二次翻转
ldp_protected_matrix = add_dp_noise_to_sparse_matrix(dp_protected_matrix, epsilon)
#生成翻转以后的噪声图
ldp_noisy_graph = nx.from_scipy_sparse_array(ldp_protected_matrix)

#原始图
print("greedypelling method")
start = datetime.now()
dense_subgraph,density = charikar_peeling(G)
# Print the nodes and edges of the dense subgraph
print('run time', datetime.now()-start, '\n')
print("Nodes in dense subgraph:", dense_subgraph.nodes())
print("Density of the dense subgraph:", density)

#差分隐私的以后的噪声图
print("greedypelling method")
start = datetime.now()
dense_subgraph_N,density_N = charikar_peeling(ldp_noisy_graph)
# Print the nodes and edges of the dense subgraph
print('run time', datetime.now()-start, '\n')
print("Nodes in dense subgraph:", dense_subgraph_N.nodes())
print("Density of the dense subgraph in LDP:", density_N)

print("Density of the dense subgraph:", density)
print("Density of the dense subgraph in LDP:", density_N)

