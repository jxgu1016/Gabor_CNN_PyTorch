import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
from .utils import FunctionBackend
from .._ext import libgcn

class GOF_Function(Function):
    @staticmethod
    def forward(ctx, weight, gaborFilterBank):
        ctx.backend = FunctionBackend(libgcn)
        ctx.backend.set_type(weight.type())
        output = weight.new()
        ctx.backend.GOF_Producing(weight, gaborFilterBank, output)
        ctx.save_for_backward(weight, gaborFilterBank)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, gaborFilterBank = ctx.saved_tensors
        grad_input = weight.new()
        ctx.backend.GOF_BPAlign(grad_input, gaborFilterBank, grad_output.data)
        return Variable(grad_input), None 

################################ OLD FASHION #############################
# class GOF_Function(Function):
#     def __init__(self):
#         super(GOF_Function, self).__init__()
#         self.backend = FunctionBackend(libgcn)

#     def forward(self, weight, gaborFilterBank):
#         output = weight.new()
#         self.backend.set_type(weight.type())
#         self.backend.GOF_Producing(weight,gaborFilterBank,output)
#         self.save_for_backward(weight, gaborFilterBank)
#         return output

#     def backward(self, grad_output):
#         weight, gaborFilterBank= self.saved_tensors
#         grad_input = weight.new()
#         self.backend.GOF_BPAlign(grad_input,gaborFilterBank,grad_output)
#         return grad_input, None
################################ OLD FASHION #############################

class my_GOF_Function(nn.Module):
    def __init__(self):
        super(my_GOF_Function, self).__init__()

    def forward(self, weight, gof):
        nOout = weight.size(0)
        nIn = weight.size(1)
        nChannel = weight.size(2)
        kH = weight.size(3)
        kW = weight.size(4)
        weight = weight.view(nOout, -1, kH, kW)
        y = []
        for i in range(gof.size(0)):
            Q = weight * gof[i]
            y.append(Q)
        index = []
        group = range(0,(nChannel-1)*nOout+1,nOout)
        for j in range(nOout):
            index.extend([k+j for k in group])
        # print(index)
        idx = torch.LongTensor(index).cuda() if weight.is_cuda else torch.LongTensor(index)
        return torch.cat(y, 0).index_select(0, Variable(idx))