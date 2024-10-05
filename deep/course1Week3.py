import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# %%
'''
TAG: 
目标：带有隐藏层的分类
构建具有隐藏层的2类分类神经网络

+ 使用具有非线性激活功能的激活函数
+ 计算损失函数
+ 实现向前向后传播
''' 


# %matplotlib inline #如果你使用用的是Jupyter Notebook的话请取消注释。

np.random.seed(1) #设置一个固定的随机种子，以保证接下来的步骤中我们的结果是一致的。


X, Y = load_planar_dataset()

# 查看数据
# X[0, :]第0行的数据
'''
print("x1的值")
print(X[0,:])
print("x2的值")
print(X[1,:])
print(Y[:]) # 色条数值
'''



# 绘图工作
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral) #绘制散点图
# 彩色背景
plt.colorbar()

# 上一语句如出现问题，请使用下面的语句：
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral) #绘制散点图
# plt.savefig('week3.png')

# X：一个numpy的矩阵，包含了这些数据点的数值
# Y：一个numpy的向量，对应着的是X的标签【0 | 1】（红色:0 ， 蓝色 :1）

shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]  # 训练集里面的数量

print ("X的维度为: " + str(shape_X))
print ("Y的维度为: " + str(shape_Y))
print ("数据集里面的数据有：" + str(m) + " 个")

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T)

# 内置绘图
plt.title("Logistic Regression") #图的标题
# plt.savefig('week3-2.png')
plot_decision_boundary(lambda x: clf.predict(x), X, Y) #绘制决策边界
# ----------进行预测
LR_predictions  = clf.predict(X.T) #预测结果
print("预测结果："+str(LR_predictions))

# J = -1/m*sum(yi*log(ai)+(1-y)log(1-ai))

# j = (Y*y+(1-Y)*(1-y))/size
print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) + 
		np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
       "% " + "(正确标记的数据点所占的百分比)")



#%%