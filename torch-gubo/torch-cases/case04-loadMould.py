import torch 
from torch.utils.data import Dataset,DataLoader,TensorDataset


class SampleDataset(Dataset):
    def __init__(self,x,y) -> None:
        """初始化"""
        self.x = x
        self.y = y
        self.len = len(y)
        

    def __len__(self):
        return self.len
    
    def __getitem__(self,idx):
        """根据索引返回一条样本"""
        idx = min(max(idx,0),self.len-1)

        return self.x[idx],self.y[idx]


def test01():

    x = torch.randn(100,8)

    y = torch.randint(0,2,[x.size(0),])

    sample_dataset = SampleDataset(x,y)
    print(sample_dataset[0])

def test02():

    x = torch.randn(100,8)

    y = torch.randint(0,2,[x.size(0),])

    sample_dataset = SampleDataset(x,y)

    dataloader = DataLoader(sample_dataset,batch_size=4,shuffle=True)

    for x,y in dataloader:
        print(x)
        print(y)
        break

def test03():

    x = torch.randn(100,8)

    y = torch.randint(0,2,[x.size(0),])

    sample_dataset = TensorDataset(x,y)
    dataloader = DataLoader(sample_dataset,batch_size=4,shuffle=True)

    for x,y in dataloader:
        print(x)
        print(y)
        break
    # print(sample_dataset[0])

if __name__ == '__main__':
    test03()