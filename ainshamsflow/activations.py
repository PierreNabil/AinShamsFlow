"""Activations Module.

In this Module, we include our activation functions
such as the Linear and Sigmoid functions.
"""

import numpy as np

from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError
#TODO: Add More Activations


def get(act_name):
	"""Get any Activation Function in this Module by name."""

	acts = [Linear, Sigmoid, Tanh, ReLU, LeakyReLU]
	for act in acts:
		if act.__name__.lower() == act_name.lower():
			return act()
	raise NameNotFoundError(act_name, __name__)


class Activation:
	"""Activation Base Class.

	To create a new Activation Function, create a class that
	inherits from this class.
	You then have to add any parameters in your constructor
	and redefine the __call__() and diff() methods.
	"""

	def __call__(self, z):
		raise BaseClassError

	def diff(self, z):
		raise BaseClassError


class Linear(Activation):
	"""Linear Activation Function."""

	__name__ = 'Linear'

	def __call__(self, z):
		return z

	def diff(self, z):
		return 1


class Sigmoid(Activation):
	"""Sigmoid Activation Function."""

	__name__ = 'Sigmoid'

	def __call__(self, z):
		return 1 / (1 + np.exp(- z))

	def diff(self, z):
		sig = self.__call__(z)
		return sig * (1 - sig)


class Tanh(Activation):
	"""Tanh Activation Function."""

	__name__ = 'Tanh'

	def __call__(self, z):
		exp_pz = np.exp(z)
		exp_nz = np.exp(-z)
		return (exp_pz - exp_nz) / (exp_pz + exp_nz)

	def diff(self, z):
		tanh = self.__call__(z)
		return 1 - np.square(tanh)


class ReLU(Activation):
	"""Rectified Linear Unit Activation Function."""

	__name__ = 'ReLU'

	def __call__(self, z):
		return np.maximum(0, z)

	def diff(self, z):
		return np.where(z > 0, 1, 0)


class LeakyReLU(Activation):
	"""Leaky ReLU Activation Function."""

	__name__ = 'LeakyRelU'

	def __init__(self, alpha=0.01):
		self.alpha = alpha

	def __call__(self, z):
		return np.maximum(self.alpha * z, z)

	def diff(self, z):
		return np.where(z > 0, 1, self.alpha)
