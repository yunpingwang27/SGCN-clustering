import networkx as nx
import random

# 节点数和社团数
num_nodes = 100
num_communities = 5

# 生成社团
community_nodes = {}
for i in range(num_communities):
    community_nodes[i] = []
    for j in range(num_nodes // num_communities):
        community_nodes[i].append(i * num_nodes // num_communities + j)

# 生成社团内部边
G = nx.Graph()
for i in range(num_communities):
    for j in range(len(community_nodes[i])):
        for k in range(j + 1, len(community_nodes[i])):
            G.add_edge(community_nodes[i][j], community_nodes[i][k])

# 生成社团间边
for i in range(num_communities):
    for j in range(i + 1, num_communities):
        for k in range(num_nodes // num_communities):
            if random.random() < 0.2:
                G.add_edge(community_nodes[i][k], community_nodes[j][k])

# 添加噪声节点和边
for i in range(num_nodes):
    G.add_node(i)
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j and random.random() < 0.01:
            G.add_edge(i, j)

# 可视化
pos = nx.spring_layout(G)
nx.draw(G, pos, node_size=50, with_labels=False)
for i in range(num_communities):
    nx.draw_networkx_nodes(G, pos, nodelist=community_nodes[i], node_color='r', node_size=50)