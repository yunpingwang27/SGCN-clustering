"""Data reading utils."""

import json
import numpy as np
import pandas as pd
from scipy import sparse
from texttable import Texttable
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, f1_score
from torch import Tensor
import torch

def read_graph(args):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers.
    :param args: Arguments object.
    :return edges: Edges dictionary.
    """
    dataset = pd.read_csv(args.edge_path).values.tolist()
    edges = {}
    edges["positive_edges"] = [edge[0:2] for edge in dataset if edge[2] == 1]
    edges["negative_edges"] = [edge[0:2] for edge in dataset if edge[2] == -1]
    edges["ecount"] = len(dataset)
    # for i in 
    # 把模块度要求的m+和m-求出来
    # edges['m_pos'] = len([edge for edge in dataset if edge[2] == 1])
    # edges['m_neg'] = edges['ecount'] - edges['m_pos']
    # edges['m_neg'] = len([edge for edge in dataset if edges[2] == -1])
    # edges['m_neg'] = len(set([edge[0] for edge in dataset if edge[2] == -1]+[edge[1] for edge in dataset if edges[2] == -1]))
    edges["ncount"] = len(set([edge[0] for edge in dataset]+[edge[1] for edge in dataset]))
    # 构建邻接矩阵
    # adj_matrix = np.zeros((edges["ncount"], edges['ncount']))
    # for edge in edges["positive_edges"]:
    #     adj_matrix[int(edge[0]), int(edge[1])] = 1
    # for edge in edges["negative_edges"]:
    #     adj_matrix[int(edge[0]), int(edge[1])] = -1
# 将邻接矩阵添加到edges字典中
    # edges["adj"] = adj_matrix.tolist()
    return edges

# import torch

def adj_matrix(positive_edges, negative_edges,):
    edges = positive_edges + negative_edges
    nodes = set()
    for edge in edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
    num_nodes = len(nodes)
    
    # 创建一个全 0 的张量作为邻接矩阵
    adj_matrix = torch.zeros(num_nodes, num_nodes)

def test_read_graph():
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers.
    :param args: Arguments object.
    :return edges: Edges dictionary.
    """
    dataset = pd.read_csv('./input/create/node_2.csv').values.tolist()
    edges = {}
    edges["positive_edges"] = [edge[0:2] for edge in dataset if edge[2] == 1]
    edges["negative_edges"] = [edge[0:2] for edge in dataset if edge[2] == -1]
    edges["ecount"] = len(dataset)
    # for i in 
    # 把模块度要求的m+和m-求出来
    edges['m_pos'] = len([edge for edge in dataset if edge[2] == 1])
    edges['m_neg'] = edges['ecount'] - edges['m_pos']
    # edges['m_neg'] = len([edge for edge in dataset if edges[2] == -1])
    # edges['m_neg'] = len(set([edge[0] for edge in dataset if edge[2] == -1]+[edge[1] for edge in dataset if edges[2] == -1]))
    edges["ncount"] = len(set([edge[0] for edge in dataset]+[edge[1] for edge in dataset]))
    return edges



def get_true_comm(args):
    # './input/create/node_comm1.csv'
    dataset = pd.read_csv(args.truecomm_path,header=None).values.tolist()
    comm = [edge[1] for edge in dataset]
    comm = torch.tensor(comm)
    return comm
# t = get_true_comm()
# print(t)

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def calculate_auc(targets, predictions, edges):
    """
    Calculate performance measures on test dataset.
    :param targets: Target vector to predict.
    :param predictions: Predictions vector.
    :param edges: Edges dictionary with number of edges etc.
    :return auc: AUC value.
    :return f1: F1-score.
    """
    targets = [0 if target == 1 else 1 for target in targets]
    auc = roc_auc_score(targets, predictions)
    pred = [1 if p > 0.5 else 0 for p in  predictions]
    f1 = f1_score(targets, pred)
    pos_ratio = sum(pred)/len(pred)
    return auc, f1, pos_ratio

def score_printer(logs):
    """
    Print the performance for every 10th epoch on the test dataset.
    :param logs: Log dictionary.
    """
    t = Texttable()
    t.add_rows([per for i, per in enumerate(logs["performance"]) if i % 10 == 0])
    print(t.draw())

def save_logs(args, logs):
    """
    Save the logs at the path.
    :param args: Arguments objects.
    :param logs: Log dictionary.
    """
    with open(args.log_path, "w") as f:
        json.dump(logs, f)

def setup_features(args, positive_edges, negative_edges, node_count):
    """
    Setting up the node features as a numpy array.
    :param args: Arguments object.
    :param positive_edges: Positive edges list.
    :param negative_edges: Negative edges list.
    :param node_count: Number of nodes.
    :return X: Node features.
    """
    if args.spectral_features:
        X = create_spectral_features(args, positive_edges, negative_edges, node_count)
    else:
        X = create_general_features(args)
    return X



def create_general_features(args):
    """
    Reading features using the path.
    :param args: Arguments object.
    :return X: Node features.
    """
    X = np.array(pd.read_csv(args.features_path))
    return X

def create_spectral_features(args, positive_edges, negative_edges, node_count):
    """
    Creating spectral node features using the train dataset edges.
    :param args: Arguments object.
    :param positive_edges: Positive edges list.
    :param negative_edges: Negative edges list.
    :param node_count: Number of nodes.
    :return X: Node features.
    """
    p_edges = positive_edges + [[edge[1], edge[0]] for edge in positive_edges]
    n_edges = negative_edges + [[edge[1], edge[0]] for edge in negative_edges]
    train_edges = p_edges + n_edges
    index_1 = [edge[0] for edge in train_edges]
    index_2 = [edge[1] for edge in train_edges]
    values = [1]*len(p_edges) + [-1]*len(n_edges)
    shaping = (node_count, node_count)
    signed_A = sparse.csr_matrix(sparse.coo_matrix((values, (index_1, index_2)),
                                                   shape=shaping,
                                                   dtype=np.float32))

    svd = TruncatedSVD(n_components=args.reduction_dimensions,
                       n_iter=args.reduction_iterations,
                       random_state=args.seed)
    svd.fit(signed_A)
    X = svd.components_.T
    return X

# below borrowed from torch_geometric
@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (Tensor, Optional[int]) -> int
    pass


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (SparseTensor, Optional[int]) -> int
    pass


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))
# def node_sampling(edge_index,num_nodes=None):
    # """choose nodes(i)"""
    # num_nodes = maybe_num_nodes(edge_index, num_nodes)

    # i = edge_index.to('cpu')
    # idx_1 = i * num_nodes + j

    # k = torch.randint(num_nodes, (i.size(0), ), dtype=torch.long)
    # idx_2 = i * num_nodes + k

    # mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
    # rest = mask.nonzero(as_tuple=False).view(-1)
    # while rest.numel() > 0:  # pragma: no cover
    #     tmp = torch.randint(num_nodes, (rest.numel(), ), dtype=torch.long)
    #     idx_2 = i[rest] * num_nodes + tmp
    #     mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
    #     k[rest] = tmp
    #     rest = rest[mask.nonzero(as_tuple=False).view(-1)]

    # return edge_index[0], edge_index[1], k.to(edge_index.device)

def get_pos_neighbor(edge_index,num_nodes = None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    i, j = edge_index.to('cpu')
    idx_1 = i * num_nodes + j
    k = torch.randint(num_nodes, (i.size(0), ), dtype=torch.long)
    idx_2 = i * num_nodes + k
    adj_list = [torch.where(edge_index[0] == i or edge_index[1] == i)[0] // num_nodes for i in range(num_nodes)]

    # mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
    # rest = mask.nonzero(as_tuple=False).view(-1)
    # while rest.numel() > 0:  # pragma: no cover
    #     tmp = torch.randint(num_nodes, (rest.numel(), ), dtype=torch.long)
    #     idx_2 = i[rest] * num_nodes + tmp
    #     mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
    #     k[rest] = tmp
    #     rest = rest[mask.nonzero(as_tuple=False).view(-1)]

    return edge_index[0], edge_index[1], k.to(edge_index.device)




def structured_negative_sampling(edge_index, num_nodes=None):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.
    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (LongTensor, LongTensor, LongTensor)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    i, j = edge_index.to('cpu')
    idx_1 = i * num_nodes + j
    # 计算每条边在一个平铺的一维数组（即张量）中的索引位置
    k = torch.randint(num_nodes, (i.size(0), ), dtype=torch.long)
    idx_2 = i * num_nodes + k

    mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.numel(), ), dtype=torch.long)
        idx_2 = i[rest] * num_nodes + tmp
        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
        k[rest] = tmp
        rest = rest[mask.nonzero(as_tuple=False).view(-1)]

    return edge_index[0], edge_index[1], k.to(edge_index.device)

def structured_negative(edge_index, num_nodes=None):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.
    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (LongTensor, LongTensor, LongTensor)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    i, j = edge_index.to('cpu')
    idx_1 = i * num_nodes + j

    k = torch.randint(num_nodes, (i.size(0), ), dtype=torch.long)
    idx_2 = i * num_nodes + k

    mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.numel(), ), dtype=torch.long)
        idx_2 = i[rest] * num_nodes + tmp
        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
        k[rest] = tmp
        rest = rest[mask.nonzero(as_tuple=False).view(-1)]

    return edge_index[0], edge_index[1], k.to(edge_index.device)



def structured_sampling(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    # i = 
    i, j = edge_index.to('cpu')
    # idx_1 = i * num_nodes + j

    # k = torch.randint(num_nodes, (i.size(0), ), dtype=torch.long)
    # idx_2 = i * num_nodes + k

    # mask = torch.from_numpy(np.isin(idx_1)).to(torch.bool)
    # rest = mask.nonzero(as_tuple=False).view(-1)
    # while rest.numel() > 0:  # pragma: no cover
        # tmp = torch.randint(num_nodes, (rest.numel(), ), dtype=torch.long)
        # idx_2 = i[rest] * num_nodes + tmp
        # mask = torch.from_numpy(np.isin(idx_1)).to(torch.bool)
        # k[rest] = tmp
        # rest = rest[mask.nonzero(as_tuple=False).view(-1)]

    return edge_index[0], edge_index[1]