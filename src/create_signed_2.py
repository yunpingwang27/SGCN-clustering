# # 可以使用Python中的NetworkX库生成带有社团结构的signed network。具体步骤如下：

# # 1.导入所需的库
import csv

import networkx as nx


import random

# 确定网络规模
n = 300  # 节点数


def creat_graph(n,pos_pro,neg_pro,in_pos,out_neg):
    communities = {}
    for i in range(n):
        if i < n//3:
            communities[i] = 0  # 社团0
        elif i < 2*(n//3) and i >= n//3:
            communities[i] = 1
        else:
            communities[i] = 2  # 社团1
    # print(communities)
    # comm = list(zip(communities,communities.values))
    # print(comm)
    with open('./input/create/node_comm500.csv', 'w', newline='') as file: 
        writer = csv.writer(file) 
        for key,value in communities.items():
            writer.writerow([key,value])
    # print(comm)
    # 构建社团内部的连边
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if communities[i] == communities[j]:
                if random.random()<pos_pro:
                    if random.random() <in_pos:
                        edges.append((i,j,1))
                    else:
                        edges.append((i,j,-1))
            else:
                if random.random() < neg_pro:
                    if random.random() <out_neg:
                        edges.append((i,j,-1))
                    else:    
                        edges.append((i,j,1))

    # 生成signed network
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # 这里报错
    G.add_weighted_edges_from(edges)
    return G,edges
G,edges = creat_graph(300,0.5,0.1,0.8,0.8)
print(G)
# 以上代码生成一个包含20个节点、50条边，带有
with open('./input/create/node_500.csv', 'w', newline='') as file: 
    writer = csv.writer(file) 
    for i in edges: 
        writer.writerow(i)






# import random
# import networkx as nx

# n = 100  # 节点个数
# m = 5  # 社团个数

# # 生成社团
# communities = [i//20 for i in range(n)]

# # 构建社团内部的连边
# edges = []
# for i in range(n):
#     for j in range(i+1, n):
#         if communities[i] == communities[j]:
#             edges.append((i, j, 1))  # 社团内部的边的权重为 1

# # 构建社团之间的连边
# for i in range(n//2):
#     for j in range(n//2, n):
#         if random.random() < 0.5:
#             edges.append((i, j, 1))  # 正权重
#         else:
#             edges.append((j, i, -1))  # 负权重

# # 生成signed network
# G = nx.Graph()
# G.add_nodes_from(range(n))
# G.add_weighted_edges_from(edges)

# # 将signed network转化为unsigned network
# unsigned_edges = []
# for i, j, w in edges:
#     if w == 1:
#         unsigned_edges.append((i, j))
#     else:
#         unsigned_edges.append((j, i))

# # 生成unsigned network
# unsigned_G = nx.Graph()
# unsigned_G.add_nodes_from(range(n))
# unsigned_G.add_edges_from(unsigned_edges)