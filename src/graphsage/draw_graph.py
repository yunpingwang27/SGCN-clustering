import matplotlib.pyplot as plt
import networkx as nx

# 创建一个空图
G = nx.Graph()

# 添加节点
G.add_nodes_from([1, 2, 3, 4, 5])

# 添加边
# G.add_edge(1, 2)
# G.add_edge(1, 3)
# G.add_edge(2, 4)
# G.add_edge(3, 4)
# G.add_edge(4, 5)
G.add_edge(1,2,weight = -1)
G.add_edge(1,3,weight = 1)
G.add_edge(1,5,weight = -1)
G.add_edge(2,3,weight = 1)
G.add_edge(3,4,weight = -1)
G.add_edge(4,5,weight = 1)


# pos = nx.spring_layout(G)
pos = nx.spring_layout(G, scale=2, k=2, iterations=100)

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=800 )

# 绘制边
# nx.draw_networkx_edges(G, pos, arrows=False)
nx.draw(G, pos, with_labels=True, node_size=800, font_color='white', font_size=16,)

# 绘制图形
# nx.draw(G, with_labels=True,node_size = 800,font_color = 'white',\
        # font_size = 16)

labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=16)

# 显示图形
plt.show()

# 