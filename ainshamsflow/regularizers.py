import numpy as np

from ainshamsflow.utils.peras_errors import BaseClassError


class Regularizer:
	def __init__(self, lambd1 = 0 , lambd2 = 0 ):
		self.lambd1 = lambd1
		self.lambd2 = lambd2

	def __call__(self, weights_list, m):
		raise BaseClassError

	def diff(self, weights_list, m):
		raise BaseClassError


class L2(Regularizer):
	def __call__(self, weights_list, m):
		return self.lambd2 * np.sum(np.square(weights_list)) / (2*m)

	def diff(self, weights_list, m):
		return self.lambd2 * np.sum(weights_list) / m


class L1(Regularizer):
	def __call__(self, weights_list, m):
		return self.lambd1 * np.sum(np.abs(weights_list)) / m

	def diff(self, weights_list, m):
		return self.lambd1 * np.sum(np.where(weights_list > 0, 1, -1)) / m
	
	
class L1L2(Regularizer):
	def __call__(self, weights_list, m):
		return self.Lambd1 * np.sum(np.abs(weights_list)) / m + self.Lambd2 * np.sum(np.square(weights_list)) / (2*m)

	def diff(self, weights_list, m):
		return self.Lambd1 * np.divide(np.where(weights_list > 0, 1, -1), m) + self.Lambd2 * np.divide(weights_list, m)
