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