# coding=utf-8
import math
import torch
# import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from utils import _pair
from torch.nn.modules.conv import _ConvNd

class MConv(_ConvNd):
	'''
	CVPR2018: Modulated Covolutional Networks
	'''
	def __init__(self, in_channels, out_channels, kernel_size, M=4, stride=1,
					padding=0, dilation=1, groups=1, bias=True, expand=False):
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(MConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
		self.expand = expand
		self.M = M
		self.MFilters = Variable(torch.stack(
								[torch.ones(kernel_size),
								torch.ones(kernel_size) * 2,
								torch.ones(kernel_size) * 3,
								torch.ones(kernel_size) * 4]))
		# print self.weight, self.bias
		# print self.MFilters

	def forward(self, x):
		if self.expand:
			x = do_expanding(x)
		if x.dim() != 5:
			raise ValueError('Experted input dim: 5,but got a {} dim\'s Tensor '.format(x.dim()))
		if x.size(1) != self.M:
			raise ValueError('Dims between MFilters:{} and input tensor:{} do not match'.format(self.M, x.size(1)))

		x_split = torch.split(x, 1, 1)
		# print x_split[0]

		y = []
		for i in range(self.M):
			Q = self.weight * self.MFilters[i]
			# print Q
			_x = x_split[i].squeeze(1)
			# print _x
			y.append(F.conv2d(_x, Q, self.bias, self.stride,
				self.padding, self.dilation, self.groups))
		return torch.stack(y, 1)

def main():
	mconv = MConv(2, 2, 3, padding=1, stride=1)
	print 'Parameters:',list(mconv.parameters())
	print 'Weight grad:',mconv.weight.grad
	raw_input = Variable(torch.randn(3,4,2,9,9))
	y = mconv(raw_input)
	print 'Output Size:', y.size()
	z = torch.mean(y)
	z.backward()
	print 'Weight grad after BP:',mconv.weight.grad


if __name__ == '__main__':
	main()