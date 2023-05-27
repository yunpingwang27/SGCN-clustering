import numpy as np

# 确定权重值域
weight_range = (-1, 1)
seed = 42

# 创建一个4x3的矩阵，并初始化为0
weights = np.zeros((4, 3))
np.random.seed(seed)
random_weights = np.random.rand(4, 3) -0.2

# 将生成的随机权重矩阵赋值给初始化好的矩阵
weights = random_weights
# weights = 
weights = np.round(random_weights, decimals=2)

# rounded_weights = [round(weight, 1) for weight in weights]
print(weights)
# np.array()
h_1 = [0.5,0.6,0.7,0.8]
h_2 = [0.3,0.8,0.3,0.4]
h_3 = [0.7,0.9,0.6,0.9]
h_4 = [0.2,0.1,0.2,0.3]
h_5 = [0.8,0.4,0.2,0.3]
w_0 = weights[:,0]
w_1 = weights[:,1]
w_2 = weights[:,2]
b = [0.05,-0.11,-0.32]
def simoid(x):
    return 1/(1+np.exp(-x))
s1,s2,s3,s4,s5 = [],[],[],[],[]
for i in range(3):
    w_i = weights[:,i]
    s1.append(round(np.dot(h_1,w_i)+b[i],2))
    s2.append(round(np.dot(h_2,w_i)+b[i],2))
    s3.append(round(np.dot(h_3,w_i)+b[i],2))
    s4.append(round(np.dot(h_4,w_i)+b[i],2))
    s5.append(round(np.dot(h_5,w_i)+b[i],2))
# import numpy as np

def softmax(x):
    exps = np.exp(x)
    exps = exps / np.sum(exps)
    exps = np.round(exps,decimals=2)
    exps = exps.tolist()
    return exps


s1 = softmax(s1)
s2 = softmax(s2)
s3 = softmax(s3)
s4 = softmax(s4)
s5 = softmax(s5)

# s = round(s,ndigits=2)
print(s1,s2,s3,s4,s5)
    # s = np.dot(h_1,w_0)
    # 
# print(np.dot(h_1,w_0))
# matrix = np.concatenate(([h_2], [h_3], [h_5]), axis=0)
# print(matrix)

grad = np.dot((s1[0]-1),h_1) + np.dot((s2[0]-1),h_2) + np.dot((s3[0]-1),h_3) +np.dot((s4[0]-1),h_4) +np.dot((s5[0]-1),h_5)
grad = np.round(grad,decimals=2)
print(grad)   

alpha = 0.2
w_0_update = w_0 - np.dot(alpha,grad)
w_0_update = np.round(w_0_update,decimals=2)
print(w_0_update)

bias = np.zeros((1, 3))
bias = np.random.rand(1, 3) * (weight_range[1] - weight_range[0]) + weight_range[0]

bias = np.round(bias, decimals=2)
print(bias)
x2 = np.add(np.dot(h_2,weights),bias)
x3 = np.add(np.dot(h_3,weights),bias)
x5 = np.add(np.dot(h_5,weights),bias)

x2 = np.round(x2, decimals=2)
x3 = np.round(x3,decimals=2)
x5 = np.round(x5,decimals=2)

x2[x2 < 0] = 0
x3[x3<0] = 0
x5[x5<0] = 0
print(x2,x3,x5)
x = np.mean([x2,x3,x5],axis=0)
x = np.round(x, decimals=2)

print(x)

t = [0.2,0.1,0.2,0.3,0.6,0.7,0.4,0.5]
# 为了使得输出是一个$3×1$的向量，我们设置$W^{(1)}$为一个 $2×8$ 的矩阵，比如我们初始化为：

w = [[1,0,0,0,1,0,1,0],[0,1,0,1,0,0,1,0]]
t = np.array(t)
w = np.array(w)
s = np.dot(w,t)
s = np.round(s,decimals=2)
norm = np.linalg.norm(s)
print(s)
s_n = s/norm
print(s_n)

l  = np.log(0.33*0.29*0.36*0.32*0.31)
l = l/5
print(l)