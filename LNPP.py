import numpy as np
import networkx as nx
from scipy.linalg import eigh, sqrtm


def laplace_noise(scale, size):
    """生成服从拉普拉斯分布的噪声"""
    return np.random.laplace(0, scale, size)


def normalize(vectors):
    """对向量进行归一化"""
    norm_vectors = []
    for v in vectors:
        norm_vectors.append(v / np.linalg.norm(vectors))
    return np.array(norm_vectors)


def orthogonalize_with_minimal_adjustment(vectors):
    """
    最小调整的向量正交化方法
    根据 Theorem 4，将非正交的向量 X 转换为正交的向量 U
    U = X C, 其中 C 是 (X^T X)^-1 的对称平方根
    """
    X = np.array(vectors)
    # 计算 (X^T X)
    XT_X = np.dot(X.T, X)

    # 检查是否为奇异矩阵，如果是奇异矩阵，返回原始向量
    if np.linalg.det(XT_X) == 0:
        print("Warning: 奇异矩阵，无法进行正交化，返回原始向量")
        return X

    # 计算 (X^T X) 的逆
    XT_X_inv = np.linalg.inv(XT_X)

    # 计算 (X^T X) 的对称平方根矩阵 C
    C = sqrtm(XT_X_inv)

    # 计算正交矩阵 U = X * C
    U = np.dot(X, C)

    return U


def compute_eigen_gap(eigenvalues, i):
    """计算特征值间隔"""
    if i == 0:
        return eigenvalues[1] - eigenvalues[0]  # 第一个特征值的间隔
    elif i == len(eigenvalues) - 1:
        return eigenvalues[-1] - eigenvalues[-2]  # 最后一个特征值的间隔
    else:
        return min(abs(eigenvalues[i] - eigenvalues[i - 1]), abs(eigenvalues[i] - eigenvalues[i + 1]))


def LNPP(A, epsilon, k):
    """
    Laplace noise calibration approach (LNPP) to satisfy ε-differential privacy.

    Parameters:
    A (numpy.ndarray): 邻接矩阵（或者图的其他矩阵表示）
    epsilon (float): 差分隐私参数
    k (int): 选择前 k 个特征值和特征向量

    Returns:
    (numpy.ndarray, numpy.ndarray): 差分隐私保护的前 k 个特征值和特征向量
    """
    # 1. 矩阵分解，获取前 k 个特征值和特征向量
    eigenvalues, eigenvectors = eigh(A)
    eigenvalues = eigenvalues[-k:]  # 取前 k 个最大的特征值
    eigenvectors = eigenvectors[:, -k:]  # 对应的特征向量

    # 2. 将 epsilon 分配给特征值和特征向量
    epsilon_values = np.full(k + 1, epsilon / (k + 1))  # 简单等分 epsilon
    epsilon_0 = epsilon_values[0]  # ε0 用于特征值
    epsilon_k = epsilon_values[1:]  # ε1,...,εk 用于特征向量

    # 3. 计算特征值的全局敏感度
    sensitivity_lambda = np.sqrt(2 * k)  # 前 k 个特征值的全局敏感度

    # 根据 Result 1，向特征值添加拉普拉斯噪声
    noisy_eigenvalues = eigenvalues + laplace_noise(scale=sensitivity_lambda / epsilon_0, size=k)

    # 4. 计算每个特征向量的全局敏感度并添加噪声
    noisy_eigenvectors = np.zeros_like(eigenvectors)
    n = A.shape[0]  # 图中的节点数量

    for i in range(k):
        # 计算第 i 个特征向量的全局敏感度，使用 Result 2
        eigen_gap = compute_eigen_gap(eigenvalues, i)  # 计算特征值间隔
        if eigen_gap == 0:
            sensitivity_u = np.inf  # 如果特征值间隔为 0，敏感度会变得非常大
        else:
            sensitivity_u = np.sqrt(n) / eigen_gap

        # 向特征向量添加拉普拉斯噪声
        noisy_eigenvectors[:, i] = eigenvectors[:, i] + laplace_noise(scale=sensitivity_u / epsilon_k[i],
                                                                      size=len(eigenvectors))

    # 5. 对生成的特征向量进行归一化和最小调整正交化
    noisy_eigenvectors = normalize(noisy_eigenvectors)
    noisy_eigenvectors = orthogonalize_with_minimal_adjustment(noisy_eigenvectors)

    # 6. 返回差分隐私保护的特征值和特征向量
    return noisy_eigenvalues, noisy_eigenvectors


# 示例使用
# 构造一个示例图的邻接矩阵
G = nx.erdos_renyi_graph(10, 0.5)
A = nx.adjacency_matrix(G).todense()

# 调用 LNPP 方法
epsilon = 1.0
k = 5
eigenvalues, eigenvectors = eigh(A)
eigenvalues = eigenvalues[-k:]  # 取前 k 个最大的特征值
eigenvectors = eigenvectors[:, -k:]  # 对应的特征向量
print("原始特征值：",eigenvalues)
print("原始特征向量",eigenvectors)
noisy_eigenvalues, noisy_eigenvectors = LNPP(A, epsilon, k)

print("差分隐私保护的特征值:", noisy_eigenvalues)
print("差分隐私保护的特征向量:", noisy_eigenvectors)
