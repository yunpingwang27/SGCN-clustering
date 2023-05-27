Skip-gram loss是一种用于训练词向量的损失函数，它的目标是最大化给定一个中心词，让其在上下文窗口内出现的其他词的概率。具体来说，skip-gram loss通常采用负对数似然损失函数，其数学表达式为：

$$
-\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j}|w_t)
$$

其中，$T$表示语料库中的总词数，$c$表示上下文窗口的大小，$w_t$表示中心词，$w_{t+j}$表示在上下文窗口内出现的其他词，$p(w_{t+j}|w_t)$表示给定中心词$w_t$，在上下文窗口内出现其他词$w_{t+j}$的概率。

通常，skip-gram loss使用负采样技术来近似计算$ p(w_{t+j}|w_t)$，即将$ p(w_{t+j}|w_t)$的计算转化为一个二分类问题，即判断$w_{t+j}$是否在上下文窗口内出现。

交叉熵函数的梯度计算是深度学习中的一个非常重要的问题，以分类问题为例，假设预测的标签是y_pred，真实的标签是y_true，交叉熵损失函数可以表示为：L(y_pred, y_true) = -sum(y_true * log(y_pred))

其中，* 表示向量的点乘，log 表示以e为底的自然对数，sum 表示向量元素的求和。这个损失函数的含义是预测标签与真实标签的差异越大，损失值越大，反之越小。

接下来，我们需要对这个损失函数求导数，求导数的结果就是梯度。首先，我们对 y_pred 求导数：

dL/dy_pred = -y_true / y_pred

然后，我们对 y_true 求导数：

dL/dy_true = -log(y_pred)

最后，我们对模型参数（例如权重和偏差）求导数时，可以利用链式法则将所有的导数相乘，得到最终的梯度值。以权重参数为例，可以表示为：

dL/dw = (dL/dy_pred) * (dy_pred/dz) * (dz/dw)

其中，z 表示输入的特征向量，dy_pred/dz 表示激活函数的导数，dz/dw 表示权重参数的导数。这个过程可以通过反向传播算法来实现。

综上所述，交叉熵损失函数的梯度计算是深度学习中非常重要的一个问题，它可以帮助我们优化模型参数，提高模型的性能。


交叉熵函数常用于分类问题的损失函数，它可以用来衡量模型输出与真实标签之间的差异。对于二分类问题，交叉熵函数的表达式如下：$$
L(y, \hat{y}) = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})
$$

其中，$y$为真实标签，$\hat{y}$为模型的预测值。接下来我们以二分类问题为例，来展示交叉熵函数对模型参数求梯度的计算过程。

假设模型有两个参数 $w_1$ 和 $w_2$，我们的目标是最小化交叉熵损失函数。为了实现这一目标，我们需要计算损失函数对每个参数的偏导数，即梯度。偏导数的计算公式如下：

$$
\frac{\partial L}{\partial w_i} = \frac{\partial}{\partial w_i}[-y\log(\hat{y}) - (1-y)\log(1-\hat{y})]
$$

我们可以将上式分解为两个部分，分别求解。

第一部分：

$$
\frac{\partial}{\partial w_i}[-y\log(\hat{y})] = -\frac{y}{\hat{y}}\frac{\partial \hat{y}}{\partial w_i}
$$

第二部分：

$$
\frac{\partial}{\partial w_i}[(1-y)\log(1-\hat{y})] = \frac{1-y}{1-\hat{y}}\frac{\partial \hat{y}}{\partial w_i}
$$

将两部分整合起来，我们可以得到损失函数对参数的梯度：

$$
\frac{\partial L}{\partial w_i} = -\frac{y}{\hat{y}}\frac{\partial \hat{y}}{\partial w_i} + \frac{1-y}{1-\hat{y}}\frac{\partial \hat{y}}{\partial w_i}
$$


Praesent in sapien. Lorem ipsum dolor sit amet, consectetuer 
adipiscing elit. Duis fringilla tristique neque. Sed interdum 
libero ut metus. Pellentesque placerat. Nam rutrum augue a leo.
Morbi sed elit sit amet ante lobortis sollicitudin.

\begin{table}[ht]
\arrayrulecolor[HTML]{DB5800}
\centering
\begin{tabular}{ |s|p{2cm}|p{2cm}|  }
\hline
\rowcolor{lightgray} \multicolumn{3}{|c|}{Country List} \\
\hline
Country Name     or Area Name& ISO ALPHA 2 Code &ISO ALPHA 3 \\
\hline
Afghanistan & AF &AFG \\
\rowcolor{gray}
Aland Islands & AX  & ALA \\
Albania    &AL & ALB \\
Algeria   &DZ & DZA \\
American Samoa & AS & ASM \\
Andorra & AD & \cellcolor[HTML]{AA0044} AND \\
Angola & AO & AGO \\
\hline
\end{tabular}
\caption{Table inside a floating element}
\label{table:ta}
\end{table}

Praesent in sapien. Lorem ipsum dolor sit amet, consectetuer 
adipiscing elit. Duis fringilla tristique neque. Sed interdum 
libero ut metus. Pellentesque placerat. Nam rutrum augue a leo. 
Morbi sed elit sit amet ante lobortis sollicitudin.
