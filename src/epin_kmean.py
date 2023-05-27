import pandas as pd
import csv
#import necessary packages
# 对嵌入表示进行聚类 
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering 
import pandas as pd
from sklearn import metrics
import sys
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score 
# args = parameter_parser()
output_path1 = 'output/epin_comm1.csv'
# output_path2 = args.outcomm_path+'2.csv'
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from com_modularity import signed_modularity

edge_list = pd.read_csv('./input/edge_list_epin.csv',header = 0)
# 读取数据
data = pd.read_csv('./output/embedding/epin.csv', header=0)
scaler = StandardScaler()
data = scaler.fit_transform(data)
data1 =data[:,1:]
s = []
# n = args.nclusters
# clustering = AgglomerativeClustering(n_clusters=n).fit(data1)
# labels1 = clustering.labels_
edge = edge_list.values.tolist()
# n = max(max(x) for x in edge) # 节点数量
# n = int(n)
# n = n+1
# n = 

# with open(output_path1, 'w', newline='') as file: 
#     writer = csv.writer(file) 
#     for index, item in enumerate(labels1): 
#         writer.writerow([index, item])
kmeans = MiniBatchKMeans(n_clusters=50,batch_size=1024).fit(data1)
labels1 = kmeans.labels_
ss = silhouette_score(data1,labels1)
print('ss score:',ss)


n = max(max(x) for x in edge) # 节点数量
n = int(n)
n = n+1
q = signed_modularity(edge_list,labels1,n)
print('Signed modularity:',q)

with open(output_path1, 'w', newline='') as file: 
    writer = csv.writer(file) 
    for index, item in enumerate(labels1): 
        writer.writerow([index, item])