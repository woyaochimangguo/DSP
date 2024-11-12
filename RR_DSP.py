import  numpy as np
from exdata import *
import numpy as np
import dsd
import networkx as nx
from dsd import *
from datetime import datetime
import matplotlib.pyplot as plt
from greedypeeling import charikar_peeling
# def generate_noisy_graph(adjacency_matrix, epsilon):
#     """
#     使用随机响应机制对给定图的邻接矩阵的下三角部分进行扰动，生成带噪声的图。
#     参数:
#     - adjacency_matrix: np.array，表示原始图的邻接矩阵 (n x n)。
#     - epsilon: float，隐私预算参数，控制差分隐私保护的强度。
#     返回:
#     - noisy_adjacency_matrix: np.array，带噪声的邻接矩阵 (n x n)。
#     """
#     n = adjacency_matrix.shape[0]
#     noisy_adjacency_matrix = adjacency_matrix.copy()
#     # 随机响应的翻转概率
#     p = 1 / (np.exp(epsilon) + 1)
#     # 对下三角部分进行扰动（i > j）
#     for i in range(1, n):
#         for j in range(i):
#             if np.random.rand() < p:
#                 # 以概率 p 翻转边的状态
#                 noisy_adjacency_matrix[i, j] = 1 - adjacency_matrix[i, j]
#                 noisy_adjacency_matrix[j, i] = noisy_adjacency_matrix[i, j]  # 确保对称性
#     return noisy_adjacency_matrix
import numpy as np


def greedy_peeling(adjacency_matrix, adjusted_density=False, epsilon=1.0):
    """
    对输入的邻接矩阵运行贪婪剥离算法，寻找最密子图，提供两种密度计算方式。

    参数:
    - adjacency_matrix: np.array，表示图的邻接矩阵 (n x n)。
    - adjusted_density: bool，是否使用调整后的密度计算方式。
    - epsilon: float，隐私预算参数，调整密度时使用。

    返回:
    - densest_subgraph: list，最密子图的节点集合。
    - max_density: float，最密子图的密度。
    """
    n = adjacency_matrix.shape[0]
    nodes = list(range(n))  # 节点集合
    degrees = np.sum(adjacency_matrix, axis=1)  # 每个节点的度
    max_density = 0
    densest_subgraph = []

    # 随机响应的翻转概率
    p = 1 / (np.exp(epsilon) + 1)
    no_flip_prob = 1 - p  # 不翻转的概率

    while len(nodes) > 0:
        # 当前子图的密度
        if adjusted_density:
            total_edges = 0
            for i in nodes:
                for j in nodes:
                    if i < j:
                        if adjacency_matrix[i, j] == 1:
                            total_edges += no_flip_prob  # 如果边存在，按不翻转概率计入
                        else:
                            total_edges += p  # 如果边不存在，按翻转概率计入
            current_density = total_edges / len(nodes)
        else:
            current_density = np.sum(degrees[nodes]) / len(nodes)

        if current_density > max_density:
            max_density = current_density
            densest_subgraph = nodes.copy()

        # 找到度最小的节点并移除
        min_degree_node = nodes[np.argmin(degrees[nodes])]
        nodes.remove(min_degree_node)
        degrees[min_degree_node] = 0  # 设为0表示节点被移除

        # 更新剩余节点的度
        for neighbor in range(n):
            if adjacency_matrix[min_degree_node, neighbor] == 1 and neighbor in nodes:
                degrees[neighbor] -= 1

    return densest_subgraph, max_density


# 生成带噪声的图
def generate_noisy_graph(adjacency_matrix, epsilon):
    n = len(adjacency_matrix)
    noisy_adjacency_matrix = adjacency_matrix.copy()

    # 随机响应的翻转概率
    p = 1 / (np.exp(epsilon) + 1)

    # 对下三角部分进行扰动（i > j）
    for i in range(1, n):
        for j in range(i):
            if np.random.rand() < p:
                # 以概率 p 翻转边的状态
                noisy_adjacency_matrix[i, j] = 1 - adjacency_matrix[i, j]
                noisy_adjacency_matrix[j, i] = noisy_adjacency_matrix[i, j]  # 确保对称性

    return noisy_adjacency_matrix


# 示例用法
if __name__ == "__main__":
    # 原始邻接矩阵 (5 个节点的无向图)
    G = nx.read_edgelist('./datasets/Facebook/facebook/414.edges', nodetype=int)

    original_adjacency_matrix = nx.adjacency_matrix(G).todense().tolist()

    epsilon = 1.0  # 隐私预算参数

    # 生成带噪声的邻接矩阵
    noisy_adjacency_matrix = generate_noisy_graph(original_adjacency_matrix, epsilon)

    # 运行贪婪剥离算法（在原始邻接矩阵上）
    densest_subgraph_original, max_density_original = greedy_peeling(original_adjacency_matrix)

    # 运行贪婪剥离算法（在带噪声的邻接矩阵上，普通计算方式）
    densest_subgraph_noisy, max_density_noisy = greedy_peeling(noisy_adjacency_matrix)

    # 运行贪婪剥离算法（在带噪声的邻接矩阵上，调整密度计算方式）
    densest_subgraph_adjusted, max_density_adjusted = greedy_peeling(noisy_adjacency_matrix, adjusted_density=True,
                                                                     epsilon=epsilon)

    # 输出结果
    print("原始邻接矩阵:")
    print(original_adjacency_matrix)
    print("\n最密子图的节点集合 (原始图):", densest_subgraph_original)
    print("最密子图的密度 (原始图):", max_density_original)

    print("\n带噪声的邻接矩阵:")
    print(noisy_adjacency_matrix)
    print("\n最密子图的节点集合 (带噪声图 - 普通计算方式):", densest_subgraph_noisy)
    print("最密子图的密度 (带噪声图 - 普通计算方式):", max_density_noisy)

    print("\n最密子图的节点集合 (带噪声图 - 概率调整方式):", densest_subgraph_adjusted)
    print("最密子图的密度 (带噪声图 - 概率调整方式):", max_density_adjusted)


# G_N = nx.from_numpy_matrix(noisy_adjacency_matrix)