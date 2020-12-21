import numpy as np

from ainshamsflow.utils.peras_errors import BaseClassError


class Regularizer:
	def __init__(self, lambd):
		self.lambd = lambd

	def __call__(self, weights_list, m):
		raise BaseClassError

	def diff(self, weights_list, m):
		raise BaseClassError


class L2(Regularizer):
	def __call__(self, weights_list, m):
		return self.lambd * np.sum(np.square(weights_list)) / (2*m)

	def diff(self, weights_list, m):
		return self.lambd * np.sum(weights_list) / m


class L1(Regularizer):
	def __call__(self, weights_list, m):
		return self.lambd * np.sum(np.abs(weights_list)) / m

	def diff(self, weights_list, m):
		return self.lambd * np.sum(np.where(weights_list > 0, 1, -1)) / m
	
	
class L1L2(Regularizer):
        def __call__(self, weights_list, m):
          regularization = backend.constant(0., dtype=x.dtype)
          if self.L1:
            regularization += self.lambd * np.sum(np.abs(weights_list)) / m
          if self.L2:
            regularization += self.lambd * np.sum(np.square(weights_list)) / (2*m)
          return regularization

        def diff(self, weights_list, m):
	  if self.L1:
          return self.lambd * np.sum(np.where(weights_list > 0, 1, -1)) / m
          if self.L2:
          return self.lambd * np.sum(weights_list) / m
