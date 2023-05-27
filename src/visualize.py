import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

# 图网络的可视化

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 读取嵌入向量数据
embeddings_df = pd.read_csv('./output/embedding/mona.csv', index_col=0)
X = embeddings_df.values   # 将数据转换为numpy数组

# 归一化数据，可选
X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# 使用t-SNE算法降维
# tsne = TSNE(n_components=2, perplexity=10, learning_rate=200)
pca = PCA(n_components=2)  # 将32维数据降到2维

X_tsne = pca.fit_transform(X_normalized)

# 从原始文件中提取节点标签属性（类别）
labels_df = pd.read_csv('./output/embedding/mona.csv',header=None)
index = labels_df.set_index(0,inplace=True)
# labels_df.set_index('id', inplace=True)
print(index)
# print(index)
# labels = labels_df.loc[embeddings_df.index].values
# print(labels)
# print(X_tsne)
# 绘制二维散点图
# X_tsne = (X_tsne - np.mean(X_tsne, axis=0)) / np.std(X_tsne, axis=0)

plt.scatter(X_tsne[:,0], X_tsne[:,1],c=index)
plt.show()


edge = pd.read_csv('input/edge_list_mona.csv',header=0).values.tolist()
# print(edge)
n = max(max(x) for x in edge)

G = nx.Graph()
G.add_nodes_from(range(n))
# 这里报错
G.add_weighted_edges_from(edge)
# for i in edges:
    # G.add_weighted_edges_from(i)
# print(G)
pos = nx.spring_layout(G)  # 定义节点的位置
labels = nx.get_edge_attributes(G, 'weight')  # 获取边的权重值

cmap = plt.cm.Reds

nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, width=1,edge_color=labels.values(), edge_cmap=cmap)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.axis('off')
plt.show()

# import csv
# 以上代码生成一个包含20个节点、50条边，带有
# with open('./input/create/node_2.csv', 'w', newline='') as file: 
#     writer = csv.writer(file) 
#     for i in edge: 
#         writer.writerow(i)



