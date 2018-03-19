# coding=utf-8
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from utils import _pair

class MConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, M=4, stride=1,
					padding=0, dilation=1, groups=1, bias=True, expand=False):
		super(MConv, self).__init__()
		if groups > 1:
			raise NotImplementedError('Group conv not implemented yet!')
		self.in_channes = in_channels
		self.out_channels = out_channels
		self.kernel_size = _pair(kernel_size)
		self.stride = _pair(stride)
		self.padding = _pair(padding)
		self.dilation = _pair(dilation)
		self.M = M
		self.expand = expand
		self.MFilters = Variable(torch.stack(
								[torch.ones(kernel_size, kernel_size),
								torch.ones(kernel_size, kernel_size) * 2,
								torch.ones(kernel_size, kernel_size) * 3,
								torch.ones(kernel_size, kernel_size) * 4]))
		print self.MFilters
		self.weight = Parameter(torch.Tensor(
				out_channels, in_channels , kernel_size, kernel_size))
		print self.weight

		if bias:
			self.bias = Parameter(torch.Tensor(M, out_channels))
		else:
			self.register_parameter('bias', None)
		print self.bias
		
	def forward(self, x):
		if self.expand:
			x = do_expanding(x)
		if x.dim() != 5:
			raise ValueError('Experted input dim: 5,but got a {} dim\'s Tensor '.format(x.dim()))
		if x.size(1) != self.M:
			raise ValueError('Dims between MFilters:{} and input tensor:{} do not match'.format(self.M, x.size(1)))

	def do_expanding(self, x):
		raise NotImplementedError

	def reset_parameters(self):
		n = self.in_channels
		for k in self.kernel_size:
			n *= k
		stdv = 1. / math.sqrt(n)
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)



def main():
	mconv = MConv(2, 2, 3)


if __name__ == '__main__':
	main()