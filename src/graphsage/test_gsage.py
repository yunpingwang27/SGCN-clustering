import numpy as np

# 定义矩阵A和B
# A = np.array([[1, 2], [3, 4]])
A = np.array([[1,0,0,0],[0,1,0,1],[0,0,1,0]])
B = np.array([[0.4],[0.6],[0.2],[0.1]])

# 计算矩阵A和B的乘积
C = np.dot(A, B)

# 显示结果
print(C)
norm = np.linalg.norm(C)
x_normalized = C / norm
print(x_normalized)