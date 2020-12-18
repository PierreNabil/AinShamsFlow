"""Initializers Module.

In this Module, we include our Initializers such as
Uniform or Normal Initializers.
"""

import numpy as np

from ainshamsflow.utils.asf_errors import BaseClassError, UnsupportedShapeError, NameNotFoundError


def get(init_name):
	"""Get any Initializer in this Module by name."""

	inits = [Constant, Uniform, Normal, Identity]
	for init in inits:
		if init.__name__.lower() == init_name.lower():
			return init()
	raise NameNotFoundError(init_name, __name__)


class Initializer:
	"""Initializer Base Class.

	To create a new Initializer, create a class that
	inherits from this class.
	You then have to add any parameters in your constructor
	and redefine the __call__() method.
	"""

	def __call__(self, shape):
		raise BaseClassError


class Constant(Initializer):
	"""Constant Value Initializer."""

	def __init__(self, value=0):
		self.value = value

	def __call__(self, shape):
		return np.full(shape, self.value)


class Uniform(Initializer):
	"""Uniform Distribution Initializer."""

	def __init__(self, start=0, end=1):
		assert start < end
		self.start = start
		self.range = end - start

	def __call__(self, shape):
		return self.range * np.random.rand(*shape) + self.start


class Normal(Initializer):
	"""Normal (Gaussian) Distribution Initializer."""

	def __init__(self, mean=0, std=1):
		assert std > 0
		self.mean = mean
		self.std = std

	def __call__(self, shape):
		return np.random.normal(self.mean, self.std, shape)


class Identity(Initializer):
	"""Identity Matrix Initializer."""

	def __init__(self, gain=1):
		self.gain = gain

	def __call__(self, shape):
		if isinstance(shape, int):
			return self.gain * np.eye(shape)
		elif len(shape) == 2:
			return self.gain * np.eye(*shape)
		else:
			raise UnsupportedShapeError(shape, 'N or (N, M)')
