from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from MConv import MConv 

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.model = nn.Sequential(

        )
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def get_network_fn(name):
    networks_zoo = {
    'gcn': GCN(),
    }
    if name is '':
        raise ValueError('Specify the network to train. All networks available:{}'.format(networks_zoo.keys()))
    elif name not in networks_zoo:
        raise ValueError('Name of network unknown {}. All networks available:{}'.format(name, networks_zoo.keys()))
    return networks_zoo[name]

def test():
    # model = TaylorNet()
    # model = TaylorNet_v2()
    # print model
    # print list(model.parameters())

    # tay = TaylorOperation(4,8)
    # print tay.alpha.data[0,0,0,0]
    # a = torch.Tensor([
    #     [
    #     [[1,1], [1,1]],
    #     [[2,2], [2,2]],
    #     [[3,3], [3,3]],
    #     [[4,4], [4,4]],
    #     ],
    #     [
    #     [[11,11], [11,11]],
    #     [[22,22], [22,22]],
    #     [[33,33], [33,33]],
    #     [[44,44], [44,44]],
    #     ],
    #     ])
    # a = Variable(a)
    # print a, a.size()
    # b = tay(a)
    # print b
    # print list(tay.parameters())

    # tay2 = TaylorOperation_v2(4)
    # print list(tay2.parameters())
    # b = tay2(a)
    # print b
    # a = Variable(torch.randn(64,3,32,32))
    # from utils import get_parameters_size
    # model = get_network_fn('inception_taylornet',gama=1.0)
    # print get_parameters_size(model)
    model = get_network_fn('gcn')
    # print get_parameters_size(model)/1e6
    # print model
    # model(a)



if __name__ == '__main__':
    test()