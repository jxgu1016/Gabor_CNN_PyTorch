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
					padding=0, dilation=1, groups=1, bias=True, expand=False):
		if groups != 1:
			raise ValueError('Group-conv not supported!')
		kernel_size = (M,) + _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(MConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
		self.expand = expand
		self.M = M
		self.need_bias = bias
		self.generate_MFilters(nScale, kernel_size)
		self.GOF_Function = GOF_Function.apply
		# self.GOF_Function = my_GOF_Function()

	def generate_MFilters(self, nScale, kernel_size):
		raise NotImplementedError

	def forward(self, x):
		if self.expand:
			x = self.do_expanding(x)
		new_weight = self.GOF_Function(self.weight, self.MFilters)
		new_bias = self.expand_bias(self.bias) if self.need_bias else self.bias
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
					padding=0, dilation=1, groups=1, bias=True, expand=False):
		super(GConv, self).__init__(in_channels, out_channels, kernel_size, M, nScale, stride,
					padding, dilation, groups, bias, expand)

	def generate_MFilters(self, nScale, kernel_size):
		# To generate Gabor Filters
		self.register_buffer('MFilters',Variable(getGaborFilterBank(nScale, *kernel_size)))

def getGaborFilterBank(nScale, M, h, w):
	Kmax = math.pi/2
	f = math.sqrt(2)
	sigma = math.pi
	sqsigma = sigma**2
	postmean = math.exp(-sqsigma/2)
	if h != 1:
		gfilter_real = torch.zeros(M, h, w)
		for i in range(M):
			theta = i / M * math.pi
			k = Kmax/f**(nScale-1)
			xymax = -1e309
			xymin = 1e309
			for y in range(h):
				for x in range(w):
					y1 = y+1-((h+1)/2)
					x1 = x+1-((w+1)/2)
					tmp1 = math.exp(-(k*k*(x1*x1+y1*y1)/(2*sqsigma)))
					tmp2 = math.cos(k*math.cos(theta)*x1+k*math.sin(theta)*y1)-postmean # For real part
					# tmp3 = math.sin(k*math.cos(theta)*x1+k*math.sin(theta)*y1) # For imaginary part
					gfilter_real[i][y][x] = k*k*tmp1*tmp2/sqsigma			
					xymax = max(xymax, gfilter_real[i][y][x])
					xymin = min(xymin, gfilter_real[i][y][x])
			gfilter_real[i] = (gfilter_real[i] - xymin) / (xymax - xymin)
	else:
		gfilter_real = torch.ones(M, h, w)
	return gfilter_real


####################################################################
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
        return torch.cat(y, 0).index_select(0, idx)