import networkx as nx
import numpy as np

# 示例图的邻接矩阵
adjacency_matrix = np.array([[0, 1, 1, 0],
                             [1, 0, 1, 1],
                             [1, 1, 0, 1],
                             [0, 1, 1, 0]])

# 创建图
G = nx.from_numpy_matrix(adjacency_matrix)

# 计算图的拉普拉斯矩阵
laplacian_matrix = nx.laplacian_matrix(G).todense()

# 计算拉普拉斯矩阵的特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

# 打印未排序的特征值和特征向量
print("未排序的特征值:")
print(eigenvalues)
print("\n未排序的特征向量:")
print(eigenvectors)

# 根据特征值从大到小排序特征向量
sorted_indices = np.argsort(eigenvalues)[::-1]  # 获取特征值排序的索引（从大到小）
sorted_eigenvalues = eigenvalues[sorted_indices]  # 排序特征值
sorted_eigenvectors = eigenvectors[:, sorted_indices]  # 排序特征向量

# 打印排序后的特征值和特征向量
print("\n排序后的特征值:")
print(sorted_eigenvalues)
print("\n排序后的特征向量:")
print(sorted_eigenvectors)
