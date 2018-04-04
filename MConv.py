# coding=utf-8
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from utils import _pair
from torch.nn.modules.conv import _ConvNd

class MConv(_ConvNd):
	'''
	Baee layer class for modulated convolution
	'''
	def __init__(self, in_channels, out_channels, kernel_size, M=4, stride=1,
					padding=0, dilation=1, groups=1, bias=True, expand=False):
		if groups != 1:
			raise ValueError('Group-conv not supported!')
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(MConv, self).__init__(
            in_channels * M, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
		self.expand = expand
		self.M = M
		self.need_bias = bias
		self.generate_MFilters(M, kernel_size)
		# print self.weight.size()
		# print self.MFilters.size()
		# print self.bias

	def generate_MFilters(self, M, kernel_size):
		self.MFilters = Parameter(torch.randn(M, *kernel_size))

	def forward(self, x):
		if self.expand:
			x = self.do_expanding(x)
		y = []	
		for i in range(self.M):
			Q = self.weight * self.MFilters[i]
			y.append(Q)
		new_weight = torch.cat(y, 0)
		# print new_weight.size()
		new_bias = torch.cat([self.bias,] * self.M, 0) if self.need_bias else self.bias
		# print new_bias
		return F.conv2d(x, new_weight, new_bias, self.stride,
				self.padding, self.dilation, self.groups)

	def do_expanding(self, x):
		_list = [x, ] * self.M # [x, x, ..., x]
		return torch.cat(_list, 1)

class GConv(MConv):
	'''
	Gabor Convolutional Operation Layer
	'''
	def __init__(self, in_channels, out_channels, kernel_size, M=4, stride=1,
					padding=0, dilation=1, groups=1, bias=True, expand=False):
		super(GConv, self).__init__(in_channels, out_channels, kernel_size, M, stride,
					padding, dilation, groups, bias, expand)
		# print self.MFilters

	def generate_MFilters(self, M, kernel_size):
		# To generate Gabor Filters
		self.MFilters = Parameter(torch.ones(M, *kernel_size))


def main():
	M = 4
	mconv = GConv(2, 3, 3, padding=1, stride=1, M=4, bias=True, groups=1)
	# mconv = MConv(2, 3, 3, padding=1, stride=1, M=4, bias=True, groups=1)
	print 'Parameters:',list(mconv.parameters())
	print 'Weight grad:',mconv.weight.grad
	raw_input = Variable(torch.ones(1, 2 * M, 6, 6))
	y = mconv(raw_input)
	print 'Output Size:', y.size()
	z = torch.mean(y)
	z.backward()
	print 'Weight grad after BP:',mconv.weight.grad
	print 'MFilters grad', mconv.MFilters.grad

def expand_test():
	mconv = MConv(2, 2, 3, padding=1, stride=1, M=4, expand=True)
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
	expand_test()
	# visulize()