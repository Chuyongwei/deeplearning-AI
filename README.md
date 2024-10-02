# [吴恩达笔记](https://www.bilibili.com/video/BV1FT4y1E74V)

准备软件包

我们需要准备一些软件包：

+ numpy：是用Python进行科学计算的基本软件包。
+ sklearn：为数据挖掘和数据分析提供的简单高效的工具。
+ matplotlib ：是一个用于在Python中绘制图表的库。
+ testCases：提供了一些测试示例来评估函数的正确性，参见下载的资料或者在底部查看它的代码。
+ planar_utils ：提供了在这个任务中使用的各种有用的功能，参见下载的资料或者在底部查看它的代码。

## week1

+ logic函数
+ sigmoid函数

## week3

### week3-1

> 手动实现深度网络的过程
 
过程：

+ 导入数据
+ 加载（处理）
+ 编写向前向后函数
+ 代价函数
+ 设计更新函数
+ 将上面的函数组合编写深度网络
+ 预测
+ 最后主函数编写，整合
+ 出图

## week4

> 目标：编写两个网络一个是2层网路，一个是多层网络篇
我们来说一下步骤：

1. 初始化网络参数
2. 前向传播
    2.1 计算一层的中线性求和的部分
    2.2 计算激活函数的部分（ReLU使用L-1次，Sigmod使用1次）
    2.3 结合线性求和与激活函数
3. 计算误差
4. 反向传播
    4.1 线性部分的反向传播公式
    4.2 激活函数部分的反向传播公式
    4.3 结合线性部分与激活函数的反向传播公式


# [pytorch笔记]( )

## tensorboared

tensorboard --logdir=logs

tensorboard --logdir=logs --port=6007

## transboard


## 卷积神经网络


```py
# nn_conv.py
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))

output = F.conv2d(input,kernel,stride=1)
```

### 导入函数

```python
import torch
from model_save import *
# 方式1-》保存方式1，加载模型
import torchvision
from torch import nn

model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(vgg16)

# 陷阱1
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model = torch.load('tudui_method1.pth')
print(model)
```

保存数据

```py
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1,模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")
```


# pytorch实战型策略

## 手写代码

### 创建数据集

```py
import numpy as np
import torch
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import matplotlib
import random

# matplotlib.use("TkAgg")
# %%
def create_dataset():
    x,y,coef = make_regression(n_samples=100,
                               n_features = 1,
                               noise =10,
                               coef=True,
                               bias = 14.5,
                               random_state=0)
    
    x = torch.tensor(x)
    y = torch.tensor(y)
    return x,y
    
def data_loader(x,y,batch_size):
    # 
    data_len = len(y)
    data_index = list(range(data_len))

    random.shuffle(data_index)

    batch_number = data_len

    for idx in range(batch_number):
        start = idx + batch_size
        end = start + batch_size

        batch_train_x = x[start:end]
        batch_train_y = y[start:end]

        yield batch_train_x,batch_train_y

def test01():

    x,y = create_dataset()
    plt.scatter(x,y)
    plt.show()

    for x,y in data_loader(x,y,batch_size=10):
        print(y)

if __name__ == "__main__":
    test01()

#%%
```

### 制作模型

```python
w = torch.tensor(0.1,device=device,requires_grad = True,dtype = torch.float64)
b = torch.tensor(0.0,device=device,requires_grad = True,dtype = torch.float64)

def linear_regression(x):
    return w*x+b

def square_loss(y_pred,y_true):
    return (y_pred-y_true) **2


def square_loss(y_pred,y_true):
    return(y_pred-y_true)**2


def sgd(lr=1e-2):
    w.data = w.data - lr*w.grad.data /16
    b.data = b.data -lr*w.grad.data / 16

```

### 训练数据

```py
def train():

    x,y,coef = create_dataset()
    epochs = 100
    learning_rate = 1e-2

    epoch_loss = []
    total_loss = 0.0
    train_samples = 0


    for _ in range(epochs):
        for train_x,train_y in data_loader(x,y,batch_size=16):

            y_pred = linear_regression(train_x)

            # print(y_pred.shape)
            # print(train_y.shape)

            loss = square_loss(y_pred,train_y.reshape(-1,1)).sum()
            total_loss += loss.item()
            train_samples += len(train_y)

            if w.grad is not None:
                w.grad.data.zero_()

            if b.grad is not None:
                b.grad.data.zero_()

            loss.backward()

            sgd(learning_rate)
            print("loss :%.10f"%(total_loss/train_samples))

        epoch_loss.append(total_loss/train_samples)

    # 绘制数据集散点图
    
    plt.scatter(x,y)

    x = torch.linspace(x.min(),x.max(),1000)
    y1 = torch.tensor([v*w+14.5 for v in x],device=device)
    y2 = torch.tensor([v*coef+14.5 for v in x],device=device)

    plt.plot(x,y1,label = "训练")
    plt.plot(x,y2,label = "真实")
    plt.grid()
    plt.title("损失变化曲线")
    plt.show()

```


