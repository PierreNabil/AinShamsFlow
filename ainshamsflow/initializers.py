import numpy as np

from ainshamsflow.utils.asf_errors import BaseClassError, UnsupportedShapeError, NameNotFoundError


def get(init_name):
	inits = [Constant, Uniform, Normal, Identity]
	for init in inits:
		if init.__name__.lower() == init_name.lower():
			return init()
	raise NameNotFoundError(init_name, __name__)


class Initializer:
	def __call__(self, shape):
		raise BaseClassError


class Constant(Initializer):
	def __init__(self, value=0):
		self.value = value

	def __call__(self, shape):
		return np.full(shape, self.value)


class Uniform(Initializer):
	def __init__(self, start=0, end=1):
		assert start < end
		self.start = start
		self.range = end - start

	def __call__(self, shape):
		return self.range * np.random.rand(*shape) + self.start


class Normal(Initializer):
	def __init__(self, mean=0, std=1):
		assert std > 0
		self.mean = mean
		self.std = std

	def __call__(self, shape):
		return np.random.normal(self.mean, self.std, shape)


class Identity(Initializer):
	def __init__(self, gain=1):
		self.gain = gain

	def __call__(self, shape):
		if isinstance(shape, int):
			return self.gain * np.eye(shape)
		elif len(shape) == 2:
			return self.gain * np.eye(*shape)
		else:
			raise UnsupportedShapeError(shape, 'N or (N, M)')
