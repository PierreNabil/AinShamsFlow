import numpy as np

from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError
#TODO: Add More Regularizers


def get(reg_name):
	regs = [L2, L1]
	for reg in regs:
		if reg.__name__.lower() == reg_name.lower():
			return reg
	else:
		raise NameNotFoundError(reg_name, __name__)


class Regularizer:
	def __init__(self, lambd):
		self.lambd = lambd

	def __call__(self, weights_list, m):
		raise BaseClassError

	def diff(self, weights_list, m):
		raise BaseClassError


class L2(Regularizer):
	__name__ = 'L2'

	def __call__(self, weights_list, m):
		return self.lambd * np.sum(np.square(weights_list)) / (2*m)

	def diff(self, weights_list, m):
		return self.lambd * np.divide(weights_list, m)


class L1(Regularizer):
	__name__ = 'L1'

	def __call__(self, weights_list, m):
		return self.lambd * np.sum(np.abs(weights_list)) / m

	def diff(self, weights_list, m):
		return self.lambd * np.divide(np.where(weights_list > 0, 1, -1), m)
