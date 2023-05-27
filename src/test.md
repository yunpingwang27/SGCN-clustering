DeepWalk算法是一种基于随机游走的无监督图表征学习方法，它可以将图中的节点映射到低维向量空间中。其主要流程如下：

1.输入：给定一个无向图$G=(V,E)$和一个随机游走的长度t，以及每个节点保留的样本数量k，以及降维后的向量维度d。

2.生成随机游走序列：对于每个节点$v\in V$，进行t次随机游走，并生成一个长度为t的序列$w_1,w_2,...,w_t$ ，其中$w_i$ 表示第i步所到达的节点。

3.使用Skip-gram模型训练词向量：将随机游走序列作为语料库，利用Skip-gram模型学习节点的向量表示。具体地，可以使用负采样来训练模型，得到每个节点的向量表示$x_v$

总结来说，DeepWalk算法的核心思路就是通过随机游走来捕捉节点间的局部信息，并通过Skip-gram模型学习节点的向量表示，从而实现无监督图表征学习。

GraphSAGE无监督学习的损失函数是负采样损失函数，其数学表达式如下：$$
L=-\sum_{i=1}^{N}\{\sum_{j\in\mathcal{N}_i}\log\sigma(\mathbf{z}_i^T\cdot\mathbf{z}_j)-k\cdot\mathbb{E}_{j\sim P_n(i)}[\log\sigma(-\mathbf{z}_i^T\cdot\mathbf{z}_j)]\}
$$

其中，$N$是节点总数，$\mathcal{N}_i$是节点$i$的邻居节点集合，$k$是负采样数量，$P_n(i)$是节点$i$的负采样分布，$\mathbf{z}_i$是节点$i$的Embedding向量，$\sigma$是sigmoid函数，$\cdot$表示向量内积。 

负采样损失函数的第一项是正样本的损失，表示节点$i$和其邻居节点$j$在Embedding空间中的内积，通过sigmoid函数将其映射为0-1之间的概率，表示$i$和$j$之间是否存在边。第二项是负样本的损失，表示$i$和一个随机采样的负样本$j$的Embedding向量内积的相反数，同样通过sigmoid函数将其映射为0-1之间的概率，表示$i$和$j$之间不存在边的概率。整个损失函数的目标是最小化正样本和负样本的损失之和。

GraphSAGE的无监督学习使用了两种损失函数：负采样损失和自监督损失。负采样损失函数：

假设对于节点 $v_i$，其邻居节点集合为 $N_i$，从负采样集合中采样得到的 $k$ 个负样本节点集合为 $N_i^-$。对于节点 $v_i$，其嵌入表示为 $z_i$，嵌入函数为 $f$。负采样损失函数的目标是最小化正样本节点与其邻居的嵌入之间的相似度，并最大化负样本节点与其邻居的嵌入之间的相似度。具体地，负采样损失函数为：

$$
\mathcal{L}_{ns}=-\sum_{i=1}^{|V|}\sum_{j\in N_i}\log\sigma(z_i^Tz_j)-\sum_{i=1}^{|V|}\sum_{j\in N_i^-}\log\sigma(-z_i^Tz_j)
$$

其中 $\sigma$ 是 sigmoid 函数。

自监督损失函数：

自监督损失函数是通过将节点嵌入表示作为输入，再通过一个 MLP（多层感知机）来预测节点的标签。具体地，对于每个节点 $v_i$，将其嵌入表示 $z_i$ 作为 MLP 的输入，预测其标签 $y_i$。自监督损失函数的目标是最小化预测标签 $y_i$ 与真实标签 $y_i^*$ 之间的差异。具体地，自监督损失函数为：

$$
\mathcal{L}_{su}=-\sum_{i=1}^{|V|}y_i^*\log\sigma(Wz_i+b)
$$

其中 $W$ 和 $b$ 是 MLP 的权重和偏置。



这段代码是用于计算Word2Vec模型的skip-gram版本中的损失函数，其中包括了两个部分：positive loss和negative loss。

数学表达式如下：

Positive Loss（true_xent）:
对于一个中心词(center_word)和其上下文词(context_word)，我们需要最大化它们条件概率的对数(log probability)，即：

log P(context_word | center_word)

使用softmax函数可以将这个条件概率转换成归一化后的概率分布，即：

P(context_word | center_word) = softmax(向量乘积(center_word, context_word))

然后，我们可以最小化交叉熵(cross-entropy)作为positive loss，即：

true_xent = - log(P(context_word | center_word))

由于Word2Vec中采用的是负采样(negative sampling)来加速训练，因此在计算negative loss时，我们需要从负样本词集合中随机采样一些词，然后计算这些词的条件概率和对应的负采样噪声(noise)的条件概率的对数值。

Negative Loss（negative_xent）:
假设我们从负样本词集合中采样得到了k个负样本词(neg_word_1, neg_word_2, ..., neg_word_k)，那么负采样的loss就是这k个词的条件概率和对应的负采样噪声的条件概率的对数值之和的相反数，即：

negative_xent = - log(1 - P(neg_word_1 | center_word)) - ... - log(1 - P(neg_word_k | center_word))

为了平衡positive loss和negative loss的重要性，我们可以对negative loss乘上一个权重系数neg_sample_weights。最终的损失函数就是两个loss之和：

loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)

其中tf.reduce_sum表示将所有元素加和起来。

