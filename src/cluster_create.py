#import necessary packages 
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering 
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from com_modularity import signed_modularity

# from ..com_modularity import signed_modularity
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_score 
edges = pd.read_csv('./input/create/node_50.csv',header=None)
# 读取数据
data = pd.read_csv('./output/embedding/create_1.csv', header=0)
# print(data[0])
# data1 =data.columns[1:]
data1 = data.iloc[:,1:]
# print(data1)
                  
s = []
# print(data1)
# print(data1.values[0])
# for i in range(2,20):
# 基于合并 的聚类算法，当只取第二项损失的时候有一个分为38
# clustering = AgglomerativeClustering(n_clusters=8).fit(data1)
    # print(clustering.labels_)
kmeans = KMeans(n_clusters=8).fit(data1)
labels = kmeans.labels_
import csv
with open('./output/k-mean/create_1.csv', 'w', newline='') as file: 
    writer = csv.writer(file) 
    for index, item in enumerate(labels): 
        writer.writerow([index, item])
ss = silhouette_score(data,labels)

modu = signed_modularity(edges,labels)
print(modu)
# 一下子0.08，这也太低了，根本不行

# ss = silhouette_score(data,labels)
# print(ss)