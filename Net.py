#cython: language_level=3

import torch.nn as nn

class Net(nn.Module):
	def __init__(self, nInput, nOutput):
		super(Net, self).__init__()
		nHidden = 1 # number of hidden neurons
		self.fca1 = nn.Linear(nInput, nHidden)
		self.fca2 = nn.Linear(nHidden, nOutput)
		#self.res = nn.Tanh()

	def forward(self, x):
		x = self.fca1(x)
		x = self.fca2(x)
		#x = self.res(x)
		return x
