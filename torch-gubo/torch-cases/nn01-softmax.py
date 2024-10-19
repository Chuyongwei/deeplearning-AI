import torch

if __name__ == '__main__':
    scores = torch.tensor([0.2,0.002,0.24,1.3,0.15,0.5,0.06,1.1,0.05,3.75])
    proba = torch.softmax(scores,dim=0)
    print(proba)