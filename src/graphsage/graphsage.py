

def sampling(src_nodes, sample_num, neighbor_table):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点

    Arguments:
    src_nodes {list, ndarray} -- 源节点列表 sample_num {int} -- 需要采样的节点数 neighbor_table {dict} -- 节点到其邻居节点的映射表

    Returns:
    ndarray -- 采样结果构成的列表
    """
    results = []
    for sid in src_nodes:
    # 从节点的邻居中进行有放回的采样
        res = np.random.choice(neighbor_table[sid], size=(sample_num, )) 
        results.append(res)
    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table): 
    """根据源节点进行多阶采样

    Arguments:
    src_nodes {list, np.ndarray} -- 源节点id sample_nums {list of int} -- 每一阶需要采样的个数 neighbor_table {dict} -- 节点到其邻居节点的映射

    Returns:
    [list of ndarray] -- 每一阶采样的结果
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table) 
        sampling_result.append(hopk_result)
    return sampling_result

class NeighborAggregator(nn.Module):
    def  init(self, input_dim, output_dim,use_bias=False, aggr_method="mean"): 
        """邻居聚合方式实现

        Arguments:

        input_dim {int} -- 输入特征的维度
        output_dim {int} -- 输出特征的维度

        Keyword Arguments:

        use_bias {bool} -- 是否使用偏置 (default: {False}) aggr_method {string} -- 聚合方式 (default: {mean})
        """
        super(NeighborAggregator, self). init () 
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim)) 
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim)) 
            self.reset_parameters()

    def reset_parameters(self): 
        init.kaiming_uniform_(self.weight) 
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature): 
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1) elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1) elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1) else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
            .format(self.aggr_method))

        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight) if self.use_bias:
        neighbor_hidden += self.bias 
        return neighbor_hidden


class GraphSage(nn.Module):
    def init(self, input_dim, hidden_dim=[64, 64], num_neighbors_list=[10, 10]):
        super(GraphSage, self).init() 
        self.input_dim = input_dim 
        self.num_neighbors_list = num_neighbors_list 
        self.num_layers = len(num_neighbors_list) 
        self.gcn = []
        self.gcn.append(SageGCN(input_dim, hidden_dim[0])) 
        self.gcn.append(SageGCN(hidden_dim[0], hidden_dim[1], activation=None))

    def forward(self, node_features_list): 
        hidden = node_features_list
        for l in range(self.num_layers): 
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l): 
                src_node_features = hidden[hop] 
                src_node_num = len(src_node_features) 
                neighbor_node_features = hidden[hop + 1] \
            .view(src_node_num, self.num_neighbors_list[hop], -1) 
                h = gcn(src_node_features, neighbor_node_features) 
                next_hidden.append(h)
            hidden = next_hidden 
        return hidden[0]

