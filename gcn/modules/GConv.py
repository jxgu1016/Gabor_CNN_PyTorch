# coding=utf-8
from __future__ import division
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from .utils import _pair
from torch.nn.modules.conv import _ConvNd
from gcn.functions import GOF_Function, my_GOF_Function

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

		# print self.weight.size()
		# print self.MFilters.size()
		# print self.bias

	def generate_MFilters(self, nScale, kernel_size):
		# self.MFilters = Parameter(torch.randn(*kernel_size))
		raise NotImplementedError

	def forward(self, x):
		if self.expand:
			x = self.do_expanding(x)
		new_weight = self.GOF_Function(self.weight, self.MFilters)
		# print new_weight.size()
		new_bias = self.expand_bias(self.bias) if self.need_bias else self.bias
		# print new_bias
		return F.conv2d(x, new_weight, new_bias, self.stride,
				self.padding, self.dilation, self.groups)

	def do_expanding(self, x):
		index = []
		for i in range(x.size(1)):
			for _ in range(self.M):
				index.append(i)
		index = torch.LongTensor(index).cuda() if x.is_cuda else torch.LongTensor(index)
		index = Variable(index)
		return x.index_select(1, index)
	
	def expand_bias(self, bias):
		index = []
		for i in range(bias.size()):
			for _ in range(self.M):
				index.append(i)
		index = torch.LongTensor(index).cuda() if x.is_cuda else torch.LongTensor(index)
		index = Variable(index)
		return bias.index_select(0, index)

class GConv(MConv):
	'''
	Gabor Convolutional Operation Layer
	'''
	def __init__(self, in_channels, out_channels, kernel_size, M=4, nScale=3, stride=1,
					padding=0, dilation=1, groups=1, bias=True, expand=False):
		super(GConv, self).__init__(in_channels, out_channels, kernel_size, M, nScale, stride,
					padding, dilation, groups, bias, expand)
		# print self.MFilters

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

def main():
	M = 4
	mconv = GConv(2, 3, 3, padding=1, stride=1, M=4, nScale=3, bias=True, groups=1).double().cuda()
	# mconv = MConv(2, 3, 3, padding=1, stride=1, M=4, bias=True, groups=1)
	print 'MFilters:', mconv.MFilters
	# print 'Parameters:',list(mconv.parameters())
	print 'Weight grad:',mconv.weight.grad
	raw_input = Variable(torch.ones(1, 2 * M, 6, 6).double().cuda())
	y = mconv(raw_input)
	print 'Output Size:', y.size()
	z = torch.mean(y)
	z.backward()
	print 'Weight grad after BP:',mconv.weight.grad
	print 'MFilters grad', mconv.MFilters.grad

def expand_test():
	mconv = GConv(2, 2, 3, padding=1, stride=1, M=4, expand=True)
	null_input = Variable(torch.Tensor([[[[1,1],[1,1]],[[2,2],[2,2]]]]))
	print null_input 
	y = mconv(null_input)
	print y

def visulize():
	from utils import visualize_graph
	from tensorboardX import SummaryWriter
	model = GConv(3, 5, 3)
	writer = SummaryWriter()
	visualize_graph(model, writer, input_size=(1, 12, 32, 32))
	writer.close()

if __name__ == '__main__':
	main()
	# expand_test()
	# visulize()