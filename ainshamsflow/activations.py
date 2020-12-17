import numpy as np

from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError
#TODO: Add More Activations


def get(act_name):
	acts = [Linear, Sigmoid, Tanh, ReLU, LeakyReLU]
	for act in acts:
		if act.__name__.lower() == act_name.lower():
			return act()
	raise NameNotFoundError(act_name, __name__)


class Activation:
	def __call__(self, z):
		raise BaseClassError

	def diff(self, z):
		raise BaseClassError


class Linear(Activation):
	__name__ = 'Linear'

	def __call__(self, z):
		return z

	def diff(self, z):
		return 1


class Sigmoid(Activation):
	__name__ = 'Sigmoid'

	def __call__(self, z):
		return 1 / (1 + np.exp(- z))

	def diff(self, z):
		sig = self.__call__(z)
		return sig * (1 - sig)


class Tanh(Activation):
	__name__ = 'Tanh'

	def __call__(self, z):
		exp_pz = np.exp(z)
		exp_nz = np.exp(-z)
		return (exp_pz - exp_nz) / (exp_pz + exp_nz)

	def diff(self, z):
		tanh = self.__call__(z)
		return 1 - np.square(tanh)


class ReLU(Activation):
	__name__ = 'ReLU'

	def __call__(self, z):
		return np.maximum(0, z)

	def diff(self, z):
		return np.where(z > 0, 1, 0)


class LeakyReLU(Activation):
	__name__ = 'LeakyRelU'

	def __init__(self, alpha=0.01):
		self.alpha = alpha

	def __call__(self, z):
		return np.maximum(self.alpha * z, z)

	def diff(self, z):
		return np.where(z > 0, 1, self.alpha)
