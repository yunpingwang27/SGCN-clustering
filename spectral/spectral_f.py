# 完整代码如下：
# 本代码将SGCN的输出特征进行谱聚类
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score 
import csv
from sklearn.decomposition import PCA
# 读取数据
from com_modularity import signed_modularity
# python src/main.py --edge-path input/ER_edges.csv --features-path input/ER_features.csv --embedding-path output/embedding/ER_edges.csv --regression-weights-path output/weights/ER_edges.csv --log-path logs/er_edges.json
import numpy as np
data = pd.read_csv('./output/embedding/ER_edges.csv', header=0)
# X = data.iloc[:, 1:].values

# 创建PCA对象，并指定要保留的主成分数量
# pca = PCA(n_components=1)

# 对数据进行主成分分析
# X_pca = pca.fit_transform(X)
# print(X_pca)
# # 删除无关列
data = data.drop(['id'], axis=1)

# # 计算相似度矩阵
similarity_matrix = cosine_similarity(data)

# # 谱聚类算法
n_clusters = 8
clusteri = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=15, assign_labels='discretize',n_jobs=-1)
clusteri.fit(similarity_matrix)
labels_1 = clusteri.labels_
data = pd.read_csv('./input/ER_edges.csv', header=0)
# def signed_modularity(data, c):

q = signed_modularity(data,labels_1)
print(q)
# # print(labels)
with open('./output/k-mean/test_spectral_sim.csv', 'w', newline='') as file: 
    writer = csv.writer(file) 
    for index, item in enumerate(labels_1): 
        writer.writerow([index, item])



