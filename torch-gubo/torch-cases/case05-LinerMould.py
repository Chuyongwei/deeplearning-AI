import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


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
    
def train():
    x,y,coef = create_dataset()
    dataset = TensorDataset(x,y)

    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)
    model = nn.Linear(in_features=1,out_features=1)

    criterion = nn.MSELoss()
    
    optimizer = optim.SGD(model.parameters(),lr=1e-2)

    epochs = 100

    for _ in range(epochs):

        for train_x,train_y in dataloader:

            # 将一个batch的训练数据送如模型

            y_pred = model(train_x.type(torch.float32))

            loss = criterion(y_pred,train_y.type(torch.float32).reshape(-1,1))
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    plt.scatter(x,y)
    x = torch.linspace(x.min(),x.max(),1000)
    y1 = torch.tensor([v*model.weight+model.bias for v in x])
    y2 = torch.tensor([v*coef+14.5 for v in x])

    plt.plot(x,y1,label="train")
    plt.plot(x,y2,label = "ture")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()

# %%