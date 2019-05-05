from __future__ import division
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
from torch.autograd.function import once_differentiable

from gcn import _C

class GOF_Function(Function):
    @staticmethod
    def forward(ctx, weight, gaborFilterBank):
        ctx.save_for_backward(weight, gaborFilterBank)
        output = _C.gof_forward(weight, gaborFilterBank)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        weight, gaborFilterBank = ctx.saved_tensors
        grad_weight = _C.gof_backward(grad_output, gaborFilterBank)
        return grad_weight, None 

class MConv(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size, M=4, nScale=3, stride=1,
                    padding=0, dilation=1, groups=1, bias=True, expand=False, padding_mode='zeros'):
        if groups != 1:
            raise ValueError('Group-conv not supported!')
        kernel_size = (M,) + _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.expand = expand
        self.M = M
        self.need_bias = bias
        self.generate_MFilters(nScale, kernel_size)
        self.GOF_Function = GOF_Function.apply

    def generate_MFilters(self, nScale, kernel_size):
        raise NotImplementedError

    def forward(self, x):
        if self.expand:
            x = self.do_expanding(x)
        new_weight = self.GOF_Function(self.weight, self.MFilters)
        new_bias = self.expand_bias(self.bias) if self.need_bias else self.bias
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            self.weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(x, new_weight, new_bias, self.stride,
                self.padding, self.dilation, self.groups)

    def do_expanding(self, x):
        index = []
        for i in range(x.size(1)):
            for _ in range(self.M):
                index.append(i)
        index = torch.LongTensor(index).cuda() if x.is_cuda else torch.LongTensor(index)
        return x.index_select(1, index)
    
    def expand_bias(self, bias):
        index = []
        for i in range(bias.size()):
            for _ in range(self.M):
                index.append(i)
        index = torch.LongTensor(index).cuda() if bias.is_cuda else torch.LongTensor(index)
        return bias.index_select(0, index)

class GConv(MConv):
    '''
    Gabor Convolutional Operation Layer
    '''
    def __init__(self, in_channels, out_channels, kernel_size, M=4, nScale=3, stride=1,
                    padding=0, dilation=1, groups=1, bias=True, expand=False, padding_mode='zeros'):
        super(GConv, self).__init__(in_channels, out_channels, kernel_size, M, nScale, stride,
                    padding, dilation, groups, bias, expand, padding_mode)

    def generate_MFilters(self, nScale, kernel_size):
        # To generate Gabor Filters
        self.register_buffer('MFilters', getGaborFilterBank(nScale, *kernel_size))

def getGaborFilterBank(nScale, M, h, w):
    Kmax = math.pi / 2
    f = math.sqrt(2)
    sigma = math.pi
    sqsigma = sigma ** 2
    postmean = math.exp(-sqsigma / 2)
    if h != 1:
        gfilter_real = torch.zeros(M, h, w)
        for i in range(M):
            theta = i / M * math.pi
            k = Kmax / f ** (nScale - 1)
            xymax = -1e309
            xymin = 1e309
            for y in range(h):
                for x in range(w):
                    y1 = y + 1 - ((h + 1) / 2)
                    x1 = x + 1 - ((w + 1) / 2)
                    tmp1 = math.exp(-(k * k * (x1 * x1 + y1 * y1) / (2 * sqsigma)))
                    tmp2 = math.cos(k * math.cos(theta) * x1 + k * math.sin(theta) * y1) - postmean # For real part
                    # tmp3 = math.sin(k*math.cos(theta)*x1+k*math.sin(theta)*y1) # For imaginary part
                    gfilter_real[i][y][x] = k * k * tmp1 * tmp2 / sqsigma			
                    xymax = max(xymax, gfilter_real[i][y][x])
                    xymin = min(xymin, gfilter_real[i][y][x])
            gfilter_real[i] = (gfilter_real[i] - xymin) / (xymax - xymin)
    else:
        gfilter_real = torch.ones(M, h, w)
    return gfilter_real