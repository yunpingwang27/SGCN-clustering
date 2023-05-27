# 本代码利用nx中的louvain进行社团划分，不考虑符号，得到的模块度也比较高
 
import pandas as pd
import csv
data = pd.read_csv('./output/embedding/bitcoin_otc_sgcn.csv', header=0)
data = data.drop(['id'], axis=1)
# print(data)
# 代码：

import pandas as pd
from sklearn.decomposition import PCA

# 读取数据
# data = pd.read_csv('./input/bitcoin_otc.csv', header=0)

# 提取除第一列以外的数据
X = data.iloc[:, 1:].values

# 创建PCA对象，并指定要保留的主成分数量
pca = PCA(n_components=1)

# 对数据进行主成分分析
X_pca = pca.fit_transform(X)

# 输出主成分分析后的数据
# print(X_pca)
# max_indices = data.idxmax(axis=1)
# print(max_indices[])

import networkx as nx
import community

# 构建图
# G = nx.karate_club_graph()

# 计算社团划分


# 输出社团划分结果
# print(partition)

data = pd.read_csv('./input/bitcoin_otc.csv',header = 0)
# print(data.iloc[:,:])
# 导入图网络输入节点的聚类结果列表和signednetwork的edgelist
# print(data[0])
edge_list = data.iloc[:,0:2]
# print(edge_list)
# 创建空网络
G = nx.Graph()
# dy = data.iloc[:,-1]


# groupby函数的使用示例：
import numpy as np
# 假设有一个DataFrame，名为df，包含一列名为“state”和一列名为“sales”：
# df = pd.DataFrame({'state':['CA','TX','NY','TX','CA'], 'sales':[1000,500,300,200,400]})
# 使用groupby函数按照state字段进行分组：
# print(comm.get_group(1))


# 添加节点
for i in range(len(X_pca)): 
    G.add_node(i,weight = X_pca[i])

# 添加边
# print(len(edge_list))
for i in range(len(edge_list)):
    source = data.iloc[i,0]
    target = data.iloc[i,1]
    flag = data.iloc[i,2]
    if flag == 1:
        G.add_edge(source,target,sign = '+')
    else:
        G.add_edge(source,target,sign = '-')
# resolution：可选，Louvain算法的分辨率参数。分辨率越高，社区划分的粒度越细。默认值为1.0。
# 66个社团
partition = community.best_partition(G,weight='weight',resolution=3)

# print(partition[1])
# d = {'a': 1, 'b': 2, 'c': 3}
nodes, comms = zip(*partition.items())
with open('./output/k-mean/test_comm.csv', 'w', newline='') as file: 
    writer = csv.writer(file) 
    for i in range(len(nodes)): 
        writer.writerow([nodes[i], comms[i]])
# print(nodes)
# 输出: ('a', 'b', 'c')
G2 = nx.Graph()
for i in range(len(edge_list)):
    source = data.iloc[i,0]
    target = data.iloc[i,1]
    flag = data.iloc[i,2]
    if flag == 1:
        G2.add_edge(source,target,sign = '+')
    else:
        G2.add_edge(source,target,sign = '-')

partition2 = community.best_partition(G2,weight='weight',resolution=3)
modularity2 = community.modularity(partition2, G2)
# 计算模块度
modularity = community.modularity(partition, G)
# G = nx.Graph()
# 打印模块度
print('Modularity:', modularity)
print('Modularity2:', modularity2)

# 计算signed network的模块度
# Modularity: 0.41387030372478495
# Modularity2: 0.4123830034844487
# Modularity: 0.41912443262489835
# Modularity2: 0.4164134939782908

# 以下是计算signed network模块度的Python代码：
# print(partition)
c = list(partition2.values())
# ```python
# 首先需要导入numpy库，计算signed network模块度的函数如下：
# print(len(data))
import numpy as np
# 符号模块度
def signed_modularity(data, c):
    n = len(set(data.iloc[:, 0]) | set(data.iloc[:, 1])) # 节点数量
    m_plus = np.zeros(n) # 与社区内节点有正边连接的节点数量
    m_minus = np.zeros(n) # 与社区内节点有负边连接的节点数量
    k_plus = np.zeros(n) # 节点的正度数
    k_minus = np.zeros(n) # 节点的负度数
    # m2_plus = 0
    # m2_minus = 0

    A = np.zeros((n, n)) # 邻接矩阵
    for i, row in data.iterrows():
        # int(row[0]) = int(int(row[0]))
        # int(row[1]) = int(int(row[1]))
        if row[2] == 1:
            A[int(row[0]), int(row[1])] += 1
            A[int(row[1]), int(row[0])] += 1
            k_plus[int(row[0])] += 1
            k_plus[int(row[1])] += 1
            # m_plus += 1
            if c[int(row[0])] == c[int(row[1])]:
                m_plus[c[int(row[0])]] += 1
        elif row[2] == -1:
            A[int(row[0]), int(row[1])] -= 1
            A[int(row[1]), int(row[0])] -= 1
            k_minus[int(row[0])] += 1
            k_minus[int(row[1])] += 1
            # m_minus += 1
            if c[int(row[0])] == c[int(row[1])]:
                m_minus[c[int(row[0])]] += 1
    Q = 0
    for i in range(n):
        for j in range(n):
            if c[i] == c[j]:
                # t_plus = 
                t_plus = (k_plus[i]*k_plus[j])/(2*m_plus.sum()) 
                t_minus = (k_minus[i]*k_minus[j])/(2*(m_minus.sum())) 
                # Q += A[i, j] - (k_plus[i]*k_plus[j]/(2*m_plus.sum()) if m_plus[c[i]]>0 else 0) + (k_minus[i]*k_minus[j]/(2*(m_minus.sum())) if m_minus[c[i]]>0 else 0)
                Q += A[i, j] - (t_plus - t_minus)
    Q /= (2*m_plus.sum() + 2*m_minus.sum())
    return Q
# return Q
signed_modu = signed_modularity(data,c)
print(signed_modu)
# 其中，m_plus、m_minus、k_plus、k_minus和A分别表示社区内节点的正边连接数量、负边连接数量、正度数、负度数和邻接矩阵。在计算过程中，先遍历边列表中的每一条边，根据边的sign值更新邻接矩阵A、节点的正度数k_plus、负度数k_minus和社区内节点的正边连接数量m_plus、负边连接数量m_minus。最后，根据上式计算signed network模块度Q。
# 其中，矩阵W是边的权重矩阵，W[i,j]表示从i到j的边的权重，取值为1或-1；k_plus是正度数数组，k_plus[i]表示节点i的正度数；k_minus是负度数数组，k_minus[i]表示节点i的负度数；m_plus是正边权重之和，m_minus是负边权重之和；Q是模块度。
    # row = [int(x) if not pd.isna(x) else 0 for x in row]
    # row = [int(x) for x in row]