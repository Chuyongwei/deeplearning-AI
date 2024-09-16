# [吴恩达笔记](https://www.bilibili.com/video/BV1FT4y1E74V)

## week1

+ logic函数
+ sigmoid函数

## week3

准备软件包
我们需要准备一些软件包：

numpy：是用Python进行科学计算的基本软件包。

sklearn：为数据挖掘和数据分析提供的简单高效的工具。

matplotlib ：是一个用于在Python中绘制图表的库。

testCases：提供了一些测试示例来评估函数的正确性，参见下载的资料或者在底部查看它的代码。

planar_utils ：提供了在这个任务中使用的各种有用的功能，参见下载的资料或者在底部查看它的代码。


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

