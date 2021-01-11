"""Activations Module.

In this Module, we include our activation functions
such as the Linear and Sigmoid functions.
"""

import numpy as np

from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError


def get(act_name):
	"""Get any Activation Function in this Module by name."""

	acts = [Linear, Sigmoid, Tanh, ReLU, LeakyReLU, SoftPlus, ELU,
			SELU, SoftSign, Swish, Softmax, HardTanh, HardSigmoid]
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


class SoftPlus(Activation):
	def __call__(self, z):
		return np.log(1 + np.exp(z))

	def diff(self, z):
		return 1 / (1 + np.exp(-z))	# exp z / 1 + exp z


class ELU(Activation):
	def __init__(self, alpha=1.67326):
		self.alpha = alpha
	
	def __call__(self, z):
		return np.where(z <= 0, self.alpha * (np.exp(z) - 1), z)
	
	def diff(self, z):
		return np.where(z < 0, self.alpha * np.exp(z), 1)


class SELU(Activation):
	def __init__(self, alpha=1.67326, lambd=1.0507):
		self.alpha = alpha
		self.lambd = lambd
	
	def __call__(self, z):
		return self.lambd * (np.where(z >= 0, z, self.alpha * (np.exp(z) - 1)))

	def diff(self, z):
		return self.lambd * (np.where(z >= 0, 1, self.alpha * (np.exp(z))))


class SoftSign(Activation):
	def __call__(self, z):
		return z / (1 + np.abs(z))
	
	def diff(self, z):
		Softsign_down = (1 + np.abs(z))
		return 1 / np.power(Softsign_down, 2)


class Swish(Activation):
	def __call__(self, z):
		return z * Sigmoid()(z)
	
	def diff(self, z):
		sig_z = Sigmoid()(z)
		return sig_z + z * sig_z * (1 - sig_z)


class Softmax(Activation):
	def __call__(self, z):
		z = z - np.max(z, axis=-1, keepdims=True)
		exp_z = np.exp(z)
		s = np.sum(exp_z, axis=-1, keepdims=True)
		ans = exp_z / s
		ans = np.where(ans == 0, 1e-6, np.where(ans > 1 - 1e-6, 1 - 1e-6, ans))
		return ans

	def diff(self, z):
		# da = np.ones(z.shape)
		# m, n = z.shape
		# p = self.__call__(z)
		# tensor1 = np.einsum('ij,ik->ijk', p, p)  # (m, n, n)
		# tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))  # (m, n, n)
		# dz = tensor2 - tensor1
		# dz = np.einsum('ijk,ik->ij', dz, da)
		# return dz
		return 1


class HardTanh(Activation):
	def __call__(self, z):
		z = np.where(z > 1, 1, z)
		z = np.where(z < -1, -1, z)
		return z

	def diff(self, z):
		return np.where(np.logical_not(np.logical_or(np.greater(z, 1), np.less(z, -1))), 1, 0)


class HardSigmoid(Activation):
	def __call__(self, z):
		return np.maximum(0, np.minimum(1, (z + 1) / 2))

	def diff(self, z):
		return np.where(np.logical_not(np.logical_or(np.less(z, -2.5), np.greater(z, 2.5)), 0.2, 0))
