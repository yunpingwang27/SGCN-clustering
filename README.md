# 符号网络的社团划分

Community Detection in Signed Network
本代码为本人本科毕业论文的配套代码；

## 面向节点聚类的符号图卷积神经网络
对SGCN进行了适用于节点聚类问题的代码调整，SGCN相关请先参考原文档`origin_README.md`。
具体实现见`src`目录。


`main.py` ----- 原SGCN；用于链接预测问题
`origin_main.py` -----原SGCN的损失函数；用于节点聚类问题（下同）
`no_neg_main.py` ----- 新损失函数1
`deep_main.py` ----- 新损失函数2
<!-- test_main.py ----- 新损失函数1 -->
以上main文件皆有对应神经网络代码，比如
```
from deep_loss_sgcn import SignedGCNTrainer
```
对应的神经网络代码即为deep_loss_sgcn.py

### 运行示例

比如，在命令行中键入

```
python src/deep_main.py --edge-path input/create/node_2.csv --embedding-path output/embedding/node_2.csv --regression-weights-path output/weights/node_2.csv --lamb 0 --epochs 50 --outcomm-path ./output/node2_comm --truecomm-path input/create/node_comm1.csv
```

其中，`--outcomm-path`指定了输出的社团划分路径，此时两种聚类方法（层次聚类与K-Means聚类）所得到的聚类结果输出为`./output/node2_comm1.csv`与`./output/node2_comm2.csv`;
`--truecomm-path`指定真实的社团划分所在路径；如未知可不指定此项。

输出的结果中包涵了对社团划分结果的评估：

```
AgglomerativeClustering Signed modularity: 0.343638589699762
KMeans Signed modularity: 0.343638589699762
AgglomerativeClustering silhouette_score: 0.8970320100239011
Kmeans silhouette_score: 0.8970320100239011
NMI score: 1.0
NMI score: 1.0
NMI score: 1.0
```
## 基于节点相似度的谱聚类算法

借助符号网络上的节点相似度进行谱聚类的社团划分策略。
代码在`src/spectral/spectral.py`
在输出社团划分结果的同时，也输出该结果的模块度评估：

```
Signed modularity: 0.31538461538461565
```

代码尚未完善，如有疏漏敬请包涵