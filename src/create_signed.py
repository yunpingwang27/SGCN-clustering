# 可以使用Python中的NetworkX库生成带有社团结构的signed network。具体步骤如下：

# 导入所需的库
import csv
import networkx as nx
import random

# 确定网络规模
n = 300  # 节点数

communities = {}
for i in range(n):
    if i < n//3:
        communities[i] = 0  # 社团0
    elif i < 2*(n//3) and i >= n//3:
        communities[i] = 1 # 社团1
    else:
        communities[i] = 2  # 社团2

# 将原社团聚类存入文件
with open('./input/create/node_comm1.csv', 'w', newline='') as file: 
    writer = csv.writer(file) 
    for key,value in communities.items():
        writer.writerow([key,value])

# 构建社团之间的连边
edges = []
for i in range(n):
    for j in range(i+1, n):
        if communities[i] == communities[j]:
            if random.random()<0.4:
                if random.random() <0.1:
                   edges.append((i,j,-1))
                else:
                    edges.append((i,j,1))
        else:
            if random.random() < 0.07:
                if random.random() <0.3:
                    edges.append((i,j,1))
                else:    
                    edges.append((i,j,-1))


#  生成signed network
G = nx.Graph()
G.add_nodes_from(range(n))
G.add_weighted_edges_from(edges)
print(G)
# 以上代码生成一个包含300个节点，有3个社团的符号网络
with open('./input/create/node_50.csv', 'w', newline='') as file: 
    writer = csv.writer(file) 
    for i in edges: 
        writer.writerow(i)

