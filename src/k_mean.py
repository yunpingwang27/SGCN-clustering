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

import os

from param_parser import parameter_parser

from sklearn.metrics import silhouette_score 
import hdbscan
import csv

# from deep_main import main

def cluster(args):
    args = parameter_parser()
    output_path1 = args.outcomm_path+'1.csv'
    output_path2 = args.outcomm_path+'2.csv'
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
    from com_modularity import signed_modularity

    edge_list = pd.read_csv(args.edge_path,header = 0)
    # 读取数据
    data = pd.read_csv(args.embedding_path, header=0)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data1 =data[:,1:]
    s = []
    n_clusters = args.nclusters
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(data1)
    labels1 = clustering.labels_
    edge = edge_list.values.tolist()
    n = max(max(x) for x in edge) # 节点数量
    n = int(n)
    n = n+1
    q = signed_modularity(edge_list,labels1,n)
    print('AgglomerativeClustering Signed modularity:',q)

    with open(output_path1, 'w', newline='') as file: 
        writer = csv.writer(file) 
        for index, item in enumerate(labels1): 
            writer.writerow([index, item])
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,batch_size=1024).fit(data1)
    # kmeans = KMeans(n_clusters=n).fit(data1)
    labels2 = kmeans.labels_
    q = signed_modularity(edge_list,labels2,n)
    print('KMeans Signed modularity:',q)
    ss1 = silhouette_score(data1,labels1)
    ss2 = silhouette_score(data1,labels2)
    print('AgglomerativeClustering silhouette_score:',ss1)
    print('Kmeans silhouette_score:',ss2)
    # nmi值
    if args.truecomm_path != None:
        comm = pd.read_csv(args.truecomm_path,header=None)
        labels_spectral = comm.iloc[:,1].to_list()
        nmi = metrics.normalized_mutual_info_score(labels1, labels_spectral)
        print("NMI score:",nmi)
        nmi = metrics.normalized_mutual_info_score(labels2, labels_spectral)
        print("NMI score:",nmi)
    else:
        pass
    nmi = metrics.normalized_mutual_info_score(labels2, labels1)

    print("NMI score:",nmi)
    labels_g = []
    for i in range(len(labels2)):
        t = []
        t.append(i)
        t.append(labels2[i])
        labels_g.append(t)
    with open(output_path2, 'w', newline='') as file: 
        writer = csv.writer(file) 
        for index, item in enumerate(labels2): 
            writer.writerow([index, item])



