# coding=utf-8
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from utils import _pair
from torch.nn.modules.conv import _ConvNd

# class MConv_2D(_ConvNd):
# 	'''
# 	'''
# 	def __init__(self, in_channels, out_channels, kernel_size, M=4, stride=1,
# 					padding=0, dilation=1, groups=1, bias=True, expand=False):
# 		kernel_size = _pair(kernel_size)
# 		stride = _pair(stride)
# 		padding = _pair(padding)
# 		dilation = _pair(dilation)
# 		super(MConv, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation,
#             False, _pair(0), groups, bias)
# 		self.expand = expand
# 		self.M = M
# 		self.MFilters = Parameter(torch.stack(
# 								[torch.ones(kernel_size),] * M))

# 		# self.MFilters = Variable(torch.stack(
# 		# 						[torch.Tensor(...),
# 		# 						...]))

# 		# print self.weight, self.bias
# 		# print self.MFilters

# 	def forward(self, x):
# 		if self.expand:
# 			x = self.do_expanding(x)
# 		if x.dim() != 5:
# 			raise ValueError('Experted input dim: 5,but got a {} dim\'s Tensor '.format(x.dim()))
# 		if x.size(1) != self.M:
# 			raise ValueError('Dims between MFilters:{} and input tensor:{} do not match'.format(self.M, x.size(1)))

# 		x_split = torch.split(x, 1, 1)
# 		# print x_split[0]

# 		y = []
# 		for i in range(self.M):
# 			Q = self.weight * self.MFilters[i]
# 			# print Q
# 			_x = x_split[i].squeeze(1)
# 			# print _x
# 			y.append(F.conv2d(_x, Q, self.bias, self.stride,
# 				self.padding, self.dilation, self.groups))
# 		return torch.stack(y, 1)

# 	def do_expanding(self, x):
# 		if x.dim() == 5:
# 			raise ValueError('No need to do expanding')
# 		_list = [x, ] * self.M
# 		# print _list
# 		# print torch.stack(_list, 1)
# 		return torch.stack(_list, 1)

class MConv(_ConvNd):
	'''
	'''
	def __init__(self, in_channels, out_channels, kernel_size, M=4, stride=1,
					padding=0, dilation=1, groups=1, bias=True, expand=False):
		kernel_size = (M, ) + _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(MConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
		self.kernel_size = kernel_size[2]
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.expand = expand
		self.M = M
		self.MFilters = Parameter(torch.stack(
								[torch.ones(kernel_size),] * M)) 
		# self.MFilters = Parameter(torch.randn(M, M, kernel_size[0], kernel_size[0]))
		# print self.weight.size()
		# print self.MFilters.size()

	def forward(self, x):
		if self.expand:
			x = self.do_expanding(x)
		y = []	
		for i in range(self.M):
			Q = self.weight * self.MFilters[i]
			y.append(Q.view(self.out_channels, self.in_channels * self.M, self.kernel_size, self.kernel_size))
		# print y
		new_weight = torch.cat(y, 0)
		# print new_weight.size()
		new_bias = torch.cat([self.bias,] * self.M, 0)
		# print new_bias
		return F.conv2d(x, new_weight, new_bias, self.stride,
				self.padding, self.dilation, self.groups)

	def do_expanding(self, x):
		_list = [x, ] * self.M # [x, x, ..., x]
		return torch.cat(_list, 1)

def main():
	M = 4
	mconv = MConv(2, 3, 3, padding=1, stride=1, M=4)
	print 'Parameters:',list(mconv.parameters())
	print 'Weight grad:',mconv.weight.grad
	raw_input = Variable(torch.ones(3, 2 * M,9,9))
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
	model = MConv(3, 5, 3)
	writer = SummaryWriter()
	visualize_graph(model, writer, input_size=(1, 4, 3, 32, 32))
	writer.close()

if __name__ == '__main__':
	main()
	# expand_test()
	# visulize()