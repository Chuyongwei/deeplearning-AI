import torch
import torch.nn as nn
import torch.optim as optim


def test01():
    # 重写了__call__方法因此可以当作函数使用
    criterion = nn.MSELoss()
    # loss = nn.MSELoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    loss = criterion(input, target)
    print(loss)

def test02():
    # 输入特征有10个
    model = nn.Linear(in_features=10,out_features=5)

    inputs = torch.randn(4,10)
    y_pred = model(inputs)
    print(y_pred.shape)

def test03():

    model = nn.Linear(in_features=10,out_features=5)
    optimizer = optim.SGD(model.parameters(),lr=1e-3)
    optimizer.zero_grad()
    optimizer.step()

if __name__ == "__main__":
    test03()

