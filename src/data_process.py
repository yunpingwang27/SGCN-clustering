import numpy as np
import pandas as pd

# A = [[0, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
#               [-1, 0, -1, -1, 0, 1, -1, 1, 0, 1, -1, -1, 1],
#               [0, -1, 0, 0, 0, 0, -1, -1, 0, 0, -1, 1, -1],
#               [-1, -1, 0, 0, 1, 1, -1, -1, 0, -1, 1, 0, -1],
#               [0, 0, 0, 1, 0, 1, -1, -1, 0, -1, 1, -1, -1],
#               [-1, 1, 0, 1, 1, 0, -1, -1, 1, -1, 1, -1, 1],
#               [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1],
#               [-1, 1, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, 1],
#               [-1, 0, 0, 0, 0, 1, -1, -1, 0, 1, 0, 1, 0],
#               [-1, 1, 0, -1, -1, -1, -1, 0, 1, 0, -1, 0, 1],
#               [-1, -1, -1, 1, 1, 1, -1, -1, 0, -1, 0, -1, -1],
#               [-1, -1, 1, 0, -1, -1, -1, 0, 1, 0, -1, 0, 0],
#               [-1, 1, -1, -1, -1, 1, -1, 1, 0, 1, -1, 0, 0]]
import csv
epinions = pd.read_table('data_signed/epinions.txt',delimiter='\t',header=None).values.tolist()
# print(epinions[0:10])
with open('edge_list_epin.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for edge in epinions:
        writer.writerow(edge)
# monastery = pd.read_table('data_signed/war1.txt',delimiter=' ',header=None).values.tolist()
monastery = pd.read_csv('data_signed/war1.csv',header=None).values.tolist()
# print(monastery)
def adj_matrix_to_edge_list(adj_matrix):
    edge_list = []
    n = len(adj_matrix)
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i][j] > 0:
                edge_list.append((i, j, 1))
            elif adj_matrix[i][j] < 0:
                edge_list.append((i, j, -1))
    return edge_list

# 示例邻接矩阵
# adj_matrix = [[0, 1, -1],
#               [1, 0, -1],
#               [-1, -1, 0]]

# 转换为边列表
edge_list = adj_matrix_to_edge_list(monastery)

# 写入CSV文件
with open('edge_list_war1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for edge in edge_list:
        writer.writerow(edge)
