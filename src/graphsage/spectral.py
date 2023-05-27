# 本代码利用符号网络的节点相似性构建相似性矩阵，利用谱聚类算法进行社团划分
# 可以考虑稀疏矩阵来进行优化提高计算速度
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score 
import csv
from scipy import sparse

# 读取数据
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from com_modularity import signed_modularity
# from com_modularity import signed_modularity
# comm = pd.read_csv('./input/create/node_comm1.csv',header=None)
# comm = pd.read_csv('./input/bitcoin_otc.csv',header=None)
# labels_true = comm.iloc[:,1].to_list()
# print(labels_true)
# data = pd.read_csv('./input/edge_list_mona.csv', header=0)
data = pd.read_csv('./input/edge_list_CV.csv', header=0)
n = len(set(data.iloc[:, 0]) | set(data.iloc[:, 1])) # 节点数量
m_plus = np.zeros(n) # 与社区内节点有正边连接的节点数量
m_minus = np.zeros(n) # 与社区内节点有负边连接的节点数量

A = np.zeros((n, n)) # 邻接矩阵
for i, row in data.iterrows():
    if row[2] == 1:
        A[int(row[0]), int(row[1])] += 1
        A[int(row[1]), int(row[0])] += 1
    elif row[2] == -1:
        A[int(row[0]), int(row[1])] -= 1
        A[int(row[1]), int(row[0])] -= 1
with open('./output/k-mean/adjacency_CV.csv', 'w', newline='') as file: 
    writer = csv.writer(file) 
    for i in A:
        writer.writerow(i)

def indices(A, i, sign):
    indice = set(filter(lambda index: A[i][index] == sign, range(len(A[i]))))
    return indice
i = 1
j = 2
def similarity(A,i,j):
    plus_i = indices(A, i, 1)
    plus_j = indices(A, j, 1)
    minus_i = indices(A, i, -1)
    minus_j = indices(A, j, -1)
    # print(minus_j)
    s_plus = len(plus_i & plus_j) + len(minus_i & minus_j)
    s_minus = len(plus_i & minus_j) +len(minus_i & plus_j)
    setiu = plus_i | minus_i
    setiu.add(i)
    setju = plus_j | minus_j
    setju.add(j)
    sxy = s_plus-s_minus + A[i][j]
    sxy /= len(setiu | setju)
    Sij = math.exp(sxy)
    # if i == j:
        # Sij = 1
    return Sij
similar = np.zeros((n, n)) # 邻接矩阵
for i in range(n):
    for j in range(i,n):
        similar[i][j] = similarity(A,i,j)
        similar[j][i] = similar[i][j]
similar = sparse.csr_matrix(similar)
# print(similar)


# 谱聚类算法
n_clusters = 3
clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=5, assign_labels='discretize')
clustering.fit(similar)

# 聚类结果
labels = clustering.labels_
q = signed_modularity(data,labels)
print('Signed modularity:',q)
# NMI（Normalized Mutual Information，标准化互信息）是一种衡量聚类算法效果的指标，它可以衡量聚类的准确性和一致性。计算NMI需要使用Python的scikit-learn库中的metrics模块。下面是一个简单的示例代码：```
from sklearn import metrics

# 假设有两个聚类结果labels_true和labels_pred
# nmi = metrics.normalized_mutual_info_score(labels, labels_true)
# print("NMI score:",nmi)
# print("NMI score is: ", nmi)
# ```

# 其中，`labels_true`是真实的类别标签，`labels_pred`是聚类算法输出的类别标签。NMI的取值范围是[0, 1]，值越大表示聚类效果越好。
# print(q)

with open('./output/spectral/test_spectral_CV.csv', 'w', newline='') as file: 
    writer = csv.writer(file) 
    for index, item in enumerate(labels): 
        writer.writerow([index, item])
