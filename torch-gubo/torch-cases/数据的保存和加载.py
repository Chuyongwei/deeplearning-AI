import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model,self).__init__()
        self.linear1 = nn.Linear(input_size,output_size*2)
        self.linear2 = nn.Linear(input_size*2,output_size)

    def forward(self,inputs):
        inputs = self.linear1(inputs)
        output = self.linear2(inputs)
        return output
    
# 实现模型参数的加载

def test1():
    model = Model(128,10)
    optimize = optim.Adam(model.parameters(),lr=1e-2)

    # 定义存储模型的参数
    save_params = {
        'init_params':{
            'input_size':128,
            'output_size':10
        },
        'acc_score':0.98,
        'avg_loss':0.86,
        'iter_num':100,
        'optim_params': optimize.state_dict(),
        'model_params': model.state_dict()
    }

    torch.save(save_params,'model/001model_params.pth')

def test2():
    model_params = torch.load('model/001model_params.pth')
    
    model = Model(model_params['init_params']['input_size'],model_params['init_params']['output_size'])
    model.load_state_dict(model_params['model_params'])
    
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(model_params['optim_params'])

    print("迭代次数",model_params['iter_num'])
    print('准确率',model_params['acc_score'])

if __name__ == '__main__':
    # test1()
    test2()