import numpy as np
import torch
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import random

# matplotlib.use("TkAgg")
# %%
# device = "cuda:0"
device = "cpu"
def create_dataset():
    x,y,coef = make_regression(n_samples=100,
                               n_features = 1,
                               noise =10,
                               coef=True,
                               bias = 14.5,
                               random_state=0)
    
    x = torch.tensor(x,device=device)
    y = torch.tensor(y,device=device)
    return x,y,coef
    
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

    plt.plot(x,y1,label = "train")
    plt.plot(x,y2,label = "gg")
    plt.grid()
    plt.title("the grid of loss")
    plt.show()




if __name__ == "__main__":
    train()

# %%