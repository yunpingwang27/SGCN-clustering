# 在这里写对输出的聚类进行模块度评估的代码？
import pandas as pd
import numpy as np
def signed_modularity(data, c,n):
    """输入的data为边列表，c为社团划分"""
    k_plus = np.zeros(n) # 节点的正度数
    k_minus = np.zeros(n) # 节点的负度数
    m_plus = 0
    m_minus = 0

    A = np.zeros((n, n)) # 邻接矩阵
    for i, row in data.iterrows():
        # int(row[0]) = int(int(row[0]))
        # int(row[1]) = int(int(row[1]))
        if row[2] == 1:
            A[int(row[0]), int(row[1])] += 1
            A[int(row[1]), int(row[0])] += 1
            k_plus[int(row[0])] += 1
            k_plus[int(row[1])] += 1
            m_plus += 1
        elif row[2] == -1:
            A[int(row[0]), int(row[1])] -= 1
            A[int(row[1]), int(row[0])] -= 1
            k_minus[int(row[0])] += 1
            k_minus[int(row[1])] += 1
            m_minus += 1
    Q = 0
    for i in range(n):
        for j in range(n):
            if c[i] == c[j]:
                t_plus = (k_plus[i]*k_plus[j])/(2*m_plus) 
                t_minus = (k_minus[i]*k_minus[j])/(2*m_minus) 
                Q += A[i, j] - (t_plus - t_minus)
    Q /= (2*m_plus + 2*m_minus)
    return Q


# q = signed_modularity(data,comm_list)
# print(q)