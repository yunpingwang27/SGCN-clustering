"""SGCN runner."""
# -----------------------
import time
import torch
import random
import numpy as np
import pandas as pd
from tqdm import trange
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from utils import calculate_auc, setup_features,get_true_comm,structured_sampling
from sklearn.model_selection import train_test_split
from signedsageconvolution import SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep
from signedsageconvolution import ListModule
from utils import structured_negative_sampling

class SignedGraphConvolutionalNetwork(torch.nn.Module):
    """
    Signed Graph Convolutional Network Class.
    For details see: Signed Graph Convolutional Network.
    Tyler Derr, Yao Ma, and Jiliang Tang ICDM, 2018.
    https://arxiv.org/abs/1808.06354
    """
    # 改了一下comm
    def __init__(self, device, args, X,comm):
        super(SignedGraphConvolutionalNetwork, self).__init__()
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        torch.manual_seed(self.args.seed)
        self.device = device
        self.X = X
        self.comm = comm
        self.setup_layers()

    def setup_layers(self):
        """
        Adding Base Layers, Deep Signed GraphSAGE layers.
        Assing Regression Parameters if the model is not a single layer model.
        """
        self.nodes = range(self.X.shape[0])
        self.neurons = self.args.layers
        self.layers = len(self.neurons)
        self.positive_base_aggregator = SignedSAGEConvolutionBase(self.X.shape[1]*2,
                                                                  self.neurons[0]).to(self.device)

        self.negative_base_aggregator = SignedSAGEConvolutionBase(self.X.shape[1]*2,
                                                                  self.neurons[0]).to(self.device)
        self.positive_aggregators = []
        self.negative_aggregators = []
        for i in range(1, self.layers):
            self.positive_aggregators.append(SignedSAGEConvolutionDeep(3*self.neurons[i-1],
                                                                       self.neurons[i]).to(self.device))

            self.negative_aggregators.append(SignedSAGEConvolutionDeep(3*self.neurons[i-1],
                                                                       self.neurons[i]).to(self.device))

        self.positive_aggregators = ListModule(*self.positive_aggregators)
        self.negative_aggregators = ListModule(*self.negative_aggregators)
        # 4改为2
        self.regression_weights = Parameter(torch.Tensor(2*self.neurons[-1], 3))
        self.regression_bias = Parameter(torch.FloatTensor(3))
        init.xavier_normal_(self.regression_weights)
        self.regression_bias.data.fill_(0.0)

    # def calculate_regression_loss(self, z, target):
    #     """
    #     Calculating the regression loss for all pairs of nodes.
    #     :param z: Hidden vertex representations.
    #     :param target: Target vector.
    #     :return loss_term: Regression loss.
    #     :return predictions_soft: Predictions for each vertex pair.
    #     """
    #     pos = torch.cat((self.positive_z_i, self.positive_z_j), 1)
    #     neg = torch.cat((self.negative_z_i, self.negative_z_j), 1)

    #     surr_neg_i = torch.cat((self.negative_z_i, self.negative_z_k), 1)
    #     surr_neg_j = torch.cat((self.negative_z_j, self.negative_z_k), 1)
    #     surr_pos_i = torch.cat((self.positive_z_i, self.positive_z_k), 1)
    #     surr_pos_j = torch.cat((self.positive_z_j, self.positive_z_k), 1)
    #     # 这里z的根本没有输入啊
    #     features = torch.cat((pos, neg, surr_neg_i, surr_neg_j, surr_pos_i, surr_pos_j))
    #     predictions = torch.mm(features, self.regression_weights) + self.regression_bias
    #     predictions_soft = F.log_softmax(predictions, dim=1)
    #     # 对此进行修改 nll指的是概率分布向量和真实标签的向量去负对数
    #     # 假设有n个样本，每个样本有k个类别，模型预测的概率分布为p，真实标签的one-hot编码为y，则nll_loss的计算公式为：
    #         # nll_loss = -1/n * sum(y * log(p))
    #     # 输出类别
    #     max_values, max_indices = torch.max(predictions_soft, dim=1)
    #     predicted_classes = max_indices.tolist()
    #     # 
    #     loss_term = F.nll_loss(predictions_soft, target)
    #     return loss_term, predictions_soft,predicted_classes
    # def calculate_community_loss(self,z,target,positive_edges,negative_edges):
        # """
        # 计算模块度损失#图全局不适合这个状况？
        # 节点相似度
        # """
        # regression_loss, self.predictions,self.clarify = self.calculate_regression_loss(z, target)
    def nll_comm_loss(self,z,target):
        """
        计算社团划分的交叉熵损失。
        :param z: Hidden vertex representation.
        :param target: Target vector.
        """
        # features = torch.cat((pos, neg, surr_neg_i, surr_neg_j, surr_pos_i, surr_pos_j))
        
        # features = torch.cat((self.node_i))
        predictions = torch.mm(z, self.regression_weights) + self.regression_bias
        predictions_soft = F.log_softmax(predictions, dim=1)
        max_values, max_indices = torch.max(predictions_soft, dim=1)
        predicted_classes = max_indices.tolist()

        # 计算余弦相似度
        # cos_sim = 
        # cos_sim = torch.nn.functional.cosine_similarity(z.unsqueeze(1), V.unsqueeze(0), dim=-1)
        # 计算相似度损失
        # sim_loss = -torch.sum(S * (y.unsqueeze(1) == y.unsqueeze(0)) * cos_sim) / (V.shape[0]**2)
        # return sim_loss
        loss_term = F.nll_loss(predictions_soft,target)
        return loss_term,predictions_soft,predicted_classes        

    # def calculate_positive_sim_loss(self, z, positive_edges,predicted_classes):
    #     i, j = structured_sampling(positive_edges,z.shape[0])
    #     self.positive_z_i = z[i]
    #     self.positive_z_j = z[j]
    #     i_list = i.tolist()
    #     j_list = j.tolist()
    #     predicted = [int(x) for x in predicted_classes]
    #     predicted = torch.tensor(predicted, dtype=torch.int64)
    #     sim_loss = 0
    #     for node_index in range(len(i_list)):
    #         if predicted[i_list[node_index]] != predicted[j_list[node_index]]:
    #             cos_sim = torch.nn.functional.cosine_similarity(z[i],z[j])
    #             if cos_sim[0]>0:
    #                 sim_loss += cos_sim[0]/(self.X.shape[0])*(self.X.shape[0])
    #             else:
    #                 sim_loss += 0
    #         else:
    #             sim_loss += 0
    #     return sim_loss

    # def calculate_negative_sim_loss(self, z, negative_edges,predicted_classes):
    #     i, j = structured_sampling(negative_edges,z.shape[0])
    #     self.positive_z_i = z[i]
    #     self.positive_z_j = z[j]
    #     i_list = i.tolist()
    #     j_list = j.tolist()
    #     predicted = [int(x) for x in predicted_classes]
    #     predicted = torch.tensor(predicted, dtype=torch.int64)
    #     sim_loss = 0
    #     for node_index in range(len(i_list)):
    #         if predicted[i_list[node_index]] == predicted[j_list[node_index]]:
    #             cos_sim = torch.nn.functional.cosine_similarity(z[i],z[j])
    #             if cos_sim[0] < 0:
    #                 sim_loss += -cos_sim[0]/(self.X.shape[0])*(self.X.shape[0])
    #             else:
    #                 sim_loss += 0
    #         else:
    #             sim_loss += 0
    #     return sim_loss
    # def calculate_sim_loss(self,z,positive_edges,negative_edges,predicted_classes):
        # edges = torch.cat(positive_edges)
    # 初始化聚类中心

    def calculate_positive_sim_loss(self, z, positive_edges, predicted_classes):
        i, j = structured_sampling(positive_edges, z.shape[0])
        positive_z_i = z[i]
        positive_z_j = z[j]
        predicted = torch.tensor(predicted_classes, dtype=torch.int64)
        cos_sim = torch.nn.functional.cosine_similarity(positive_z_i, positive_z_j)
        sim_loss = torch.where(predicted[i] != predicted[j], torch.clamp(cos_sim, min=0), torch.tensor(0.))
        sim_loss = torch.mean(sim_loss)
        return sim_loss
    
    def calculate_regression_loss(self, z, target):
        """
        Calculating the regression loss for all pairs of nodes.
        :param z: Hidden vertex representations.
        :param target: Target vector.
        :return loss_term: Regression loss.
        :return predictions_soft: Predictions for each vertex pair.
        """
        pos = torch.cat((self.positive_z_i, self.positive_z_j), 1)
        neg = torch.cat((self.negative_z_i, self.negative_z_j), 1)

        surr_neg_i = torch.cat((self.negative_z_i, self.negative_z_k), 1)
        surr_neg_j = torch.cat((self.negative_z_j, self.negative_z_k), 1)
        surr_pos_i = torch.cat((self.positive_z_i, self.positive_z_k), 1)
        surr_pos_j = torch.cat((self.positive_z_j, self.positive_z_k), 1)

        features = torch.cat((pos, neg, surr_neg_i, surr_neg_j, surr_pos_i, surr_pos_j))
        predictions = torch.mm(features, self.regression_weights) + self.regression_bias
        predictions_soft = F.log_softmax(predictions, dim=1)
        # 对此进行修改 nll指的是概率分布向量和真实标签的向量去负对数
        # 假设有n个样本，每个样本有k个类别，模型预测的概率分布为p，真实标签的one-hot编码为y，则nll_loss的计算公式为：
            # nll_loss = -1/n * sum(y * log(p))
        loss_term = F.nll_loss(predictions_soft, target)
        return loss_term, predictions_soft

    
    
    # 可以尝试使用向量化运算来优化这段代码，以减少循环次数，加速计算速度。其中可以使用torch.where()函数来实现条件筛选。具体实现如下：```python
    def calculate_negative_sim_loss(self, z, negative_edges, predicted_classes):
        i, j = structured_sampling(negative_edges, z.shape[0])
        positive_z_i = z[i]
        positive_z_j = z[j]
        # out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        predicted = torch.tensor(predicted_classes, dtype=torch.int64)
        cos_sim = torch.nn.functional.cosine_similarity(positive_z_i, positive_z_j)
        sim_loss = torch.where(predicted[i] == predicted[j], -torch.clamp(cos_sim, max=0), torch.tensor(0.))
        sim_loss = torch.mean(sim_loss)
        return sim_loss
    # def calculate_nolink_sim_loss(self,z,positive_edges,,predicted_classes):

   
    # def calculate_loss_function(self, z, positive_edges, negative_edges, target):
    def calculate_loss_function(self, z,positive_edges,negative_edges,target):
        """
        Calculating the embedding losses, regression loss and weight regularization loss.
        :param z: Node embedding.
        :param positive_edges: Positive edge pairs.
        :param negative_edges: Negative edge pairs.
        :param target: Target vector.
        :return loss: Value of loss.
        """
        regression_loss, self.predictions,self.clarify = self.nll_comm_loss(z, target)
        # embedding_pos = self.calculate_positive_embedding_loss(z,positive_edges)
        # embedding_neg = self.calculate_negative_embedding_loss(z,negative_edges)
        sim_loss1 = self.calculate_positive_sim_loss(z,positive_edges,self.clarify)
        sim_loss2 = self.calculate_negative_sim_loss(z,negative_edges,self.clarify)
        # loss,regress = self.calculate_regression_loss(z,target)
        # sim_loss = self.calculate_sim_loss(z,positive_edges,negative_edges,self.clarify)
        # loss_term = regression_loss +self.args.lamb*(sim_loss+sim_loss2)
        # loss_term = self.args.lamb*(regression_loss) +(1-self.args.lamb)*(sim_loss1+sim_loss2)+loss
        loss_term = self.args.lamb*(regression_loss)+(1-self.args.lamb)*(sim_loss1+sim_loss2)
        return loss_term,self.clarify

    def forward(self, positive_edges, negative_edges,comm):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        :param positive_edges: Positive edges.
        :param negative_edges: Negative edges.
        :param target: Target vectors.
        :return loss: Loss value.
        :return self.z: Hidden vertex representations.
        """
        self.h_pos, self.h_neg = [], []
        self.h_pos.append(torch.tanh(self.positive_base_aggregator(self.X, positive_edges)))
        self.h_neg.append(torch.tanh(self.negative_base_aggregator(self.X, negative_edges)))
        for i in range(1, self.layers):
            self.h_pos.append(torch.tanh(self.positive_aggregators[i-1](self.h_pos[i-1], self.h_neg[i-1], positive_edges, negative_edges)))
            self.h_neg.append(torch.tanh(self.negative_aggregators[i-1](self.h_neg[i-1], self.h_pos[i-1], positive_edges, negative_edges)))
        self.z = torch.cat((self.h_pos[-1], self.h_neg[-1]), 1)
        loss,clarify = self.calculate_loss_function(self.z,positive_edges,negative_edges, comm)
        return loss, self.z,clarify

class SignedGCNTrainer(object):
    """
    Object to train and score the SGCN, log the model behaviour and save the output.
    """
    def __init__(self, args, edges,comm):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param edges: Edge data structure with positive and negative edges separated.
        """
        self.args = args
        self.edges = edges
        # self.test_edges = test_edges
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logs()
        self.comm = comm

    def setup_logs(self):
        """
        Creating a log dictionary.
        """
        self.logs = {}
        self.logs["parameters"] = vars(self.args)
        self.logs["performance"] = [["Epoch", "AUC", "F1", "pos_ratio"]]
        self.logs["training_time"] = [["Epoch", "Seconds"]]

    def setup_dataset(self):
        """
        Creating train and test split.
        """
    
        self.positive_edges, self.test_positive_edges = train_test_split(self.edges["positive_edges"],
                                                                         test_size=self.args.test_size)

        self.negative_edges, self.test_negative_edges = train_test_split(self.edges["negative_edges"],
                                                                         test_size=self.args.test_size)
        # self.positive_edges = self.edges['positive_edges']
        # self.negative_edges = self.edges['negative_edges']
        # self.test_positive_edges = self.test_edges['positive_edges']
        # self.test_negative_edges = self.test_edges['negative_edges']    
        self.ecount = len(self.positive_edges + self.negative_edges)

        self.X = setup_features(self.args,
                                self.positive_edges,
                                self.negative_edges,
                                self.edges["ncount"])
        self.positive_edges = torch.from_numpy(np.array(self.positive_edges,
                                                        dtype=np.int64).T).type(torch.long).to(self.device)

        self.negative_edges = torch.from_numpy(np.array(self.negative_edges,
                                                        dtype=np.int64).T).type(torch.long).to(self.device)
        
        # self.y1 = np.array([0 if i < int(self.ecount/2) else 1 for i in range(self.ecount)]+[2]*(self.ecount*2))
        self.y = np.array(self.comm)
        # self.y = np.array([0]*100+[1]*100+[2]*100)
        # self.y = np.array(self.y)
        # self.y1 = torch.from_numpy(self.y).type(torch.LongTensor).to(self.device)
        self.y = torch.from_numpy(self.y).type(torch.LongTensor).to(self.device)
        self.X = torch.from_numpy(self.X).float().to(self.device)

    def score_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        :param epoch: Epoch number.
        """
        # 原来的话是把两节点之间的链接分为正、负和无
        # 需要修改为已知社团划分
        # 原来的话，是本来这些图的节点都在这？然后分训练集测试集，拆的只是边
        # 现在的话，是没法这样的（我指test_z根据train_z得出什么的？）
        # 我还有一个问题，就是这个分类器是怎么进行多分类的？就是得出的这个权重，怎么代进去进行多分类之类的
        # 目前来讲，难道说我们要拆一部分图的节点么？这样的话，网络肯定是没法输入正边和负边的；看来也许只能整体进行测试？
        # 但是怎么代入模型？根据那个score_mm那两句

        loss, self.train_z,clarify = self.model(self.positive_edges, self.negative_edges,self.comm)
        self.clarify = clarify
        # test_edges = torch.cat((self.test_positive_edges,self.test_negative_edges))
        test_z = self.train_z
        # score_positive_edges = torch.from_numpy(np.array(self.test_positive_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        # score_negative_edges = torch.from_numpy(np.array(self.test_negative_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        # test_positive_z = torch.cat((self.train_z[score_positive_edges[0, :], :], self.train_z[score_positive_edges[1, :], :]), 1)
        # test_negative_z = torch.cat((self.train_z[score_negative_edges[0, :], :], self.train_z[score_negative_edges[1, :], :]), 1)
        scores = torch.mm(test_z, self.model.regression_weights.to(self.device)) + self.model.regression_bias.to(self.device)
        probability_scores = torch.exp(F.softmax(scores, dim=1))
        predictions = probability_scores[:, 0]/probability_scores[:, 0:2].sum(1)
        predictions = predictions.cpu().detach().numpy()
        targets = self.comm
        # targets = [0]*100 + [1]*100 + [2]*100
        targets1 = [0]*len(self.test_positive_edges) + [1]*len(self.test_negative_edges)
        auc, f1, pos_ratio = calculate_auc(targets, predictions, self.edges)
        self.logs["performance"].append([epoch+1, auc, f1, pos_ratio])

        # loss, self.train_z,clarify = self.model(self.positive_edges, self.negative_edges,self.comm)
        # self.clarify = clarify
        # score_positive_edges = torch.from_numpy(np.array(self.test_positive_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        # score_negative_edges = torch.from_numpy(np.array(self.test_negative_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        # test_z = self.train_z
        # # test_positive_z = self.train_z
        # # test_negative_z =

        # score_nodes = torch.from_numpy(np.array(self.comm,dtype=np.int64).T).type(torch.long).to(self.device)
        # ones = torch.ones(1,300).to(self.device)
        # # test_z = torch.cat(self.train_z)
        # train_z = self.train_z

        # scores = torch.mm(torch.cat((test_positive_z, test_negative_z), 0), self.model.regression_weights.to(self.device)) + self.model.regression_bias.to(self.device)

        # score_comm = torch.mm(train_z,self.model.regression_weights.to(self.device))+ self.model.regression_bias.to(self.device)

        # probability_scores = torch.exp(F.softmax(score_comm, dim=1))
        # # 为什么是0:2
        # predictions = probability_scores[:, 0]/probability_scores[:, 0:2].sum(1)
        
        # predictions = predictions.cpu().detach().numpy()
        
        # targets = [0]*100 + [1]*100 + [2]*100
        # auc, f1, pos_ratio = calculate_auc(targets, predictions, self.edges)
        # self.logs["performance"].append([epoch+1, auc, f1, pos_ratio])

    def create_and_train_model(self):
        """
        Model training and scoring.
        """
        print("\nTraining started.\n")
        self.model = SignedGraphConvolutionalNetwork(self.device, self.args,self.X, self.comm).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.model.train()
        self.epochs = trange(self.args.epochs, desc="Loss")
        for epoch in self.epochs:
            start_time = time.time()
            self.optimizer.zero_grad()
            loss, _ ,_= self.model(self.positive_edges,self.negative_edges, self.comm)
            loss.backward()
            self.epochs.set_description("SGCN (Loss=%g)" % round(loss.item(), 4))
            self.optimizer.step()
            self.logs["training_time"].append([epoch+1, time.time()-start_time])
            if self.args.test_size > 0:
                self.score_model(epoch)

    def save_model(self):
        """
        Saving the embedding and model weights.
        """
        print("\nEmbedding is saved.\n")
        self.train_z = self.train_z.cpu().detach().numpy()
        embedding_header = ["id"] + ["x_"+str(x) for x in range(self.train_z.shape[1])]
        self.train_z = np.concatenate([np.array(range(self.train_z.shape[0])).reshape(-1, 1), self.train_z], axis=1)
        self.train_z = pd.DataFrame(self.train_z, columns=embedding_header)
        self.train_z.to_csv(self.args.embedding_path, index=None)
        print("\nRegression parameters are saved.\n")
        self.regression_weights = self.model.regression_weights.cpu().detach().numpy().T
        self.regression_bias = self.model.regression_bias.cpu().detach().numpy().reshape((3, 1))
        regression_header = ["x_" + str(x) for x in range(self.regression_weights.shape[1])] + ["bias"]
        self.regression_params = pd.DataFrame(np.concatenate((self.regression_weights, self.regression_bias), axis=1), columns=regression_header)
        self.regression_params.to_csv(self.args.regression_weights_path, index=None)
        print("\nRegression results are saved.\n")
        # self.comm = self.model.regression_weights.cpu().detach().numpy().T
        # self.regression_bias = self.model.regression_bias.cpu().detach().numpy().reshape((3, 1))
        # regression_header = ["x_" + str(x) for x in range(self.regression_weights.shape[1])] + ["bias"]
        regression_re_header = ['comm']
        self.regression_re = pd.DataFrame(self.clarify,columns=regression_re_header)
        self.regression_re.to_csv('output/comm.csv', index=None)
        
# 0,299？
# 就算是边数也该是101