import numpy as np

from ainshamsflow import activations
from ainshamsflow import initializers
from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError
#TODO: Add More Layers


def get(layer_name):
	layers = [Dense, BatchNorm, Dropout,
			  Conv1D, Conv2D, AvgPool, MaxPool, GlobalAvgPool, GlobalMaxPool, Flatten, Upsample1D, Upsample2D,
			  Lambda, Activation, Add, Concat, Reshape]
	for layer in layers:
		if layer.__name__.lower() == layer_name.lower():
			return layer
	else:
		raise NameNotFoundError(layer_name, __name__)


class Layer:
	def __init__(self, name, trainable):
		assert isinstance(trainable, bool)
		assert isinstance(name, str)

		self.trainable = trainable
		self.name = name

	def __call__(self, x, training=False):
		raise BaseClassError

	def diff(self, prev_da):
		raise BaseClassError

	def count_params(self):
		raise BaseClassError

	def get_weights(self):
		raise BaseClassError

	def set_weights(self, weights, biases):
		raise BaseClassError

	def summary(self) :
		return '{:20s} | {:13d} | {:>21} | {:>21} | {:30}'.format(self.__name__ + ' Layer:', self.count_params(),
													   self.input_shape, self.output_shape, self.name)

# DNN Layers:


class Dense(Layer):
	__name__ = 'Dense'

	def __init__(self, n_out, name, activation=activations.Linear(),
				 weights_init=initializers.Normal(), biases_init=initializers.Constant(0),
				 trainable=True):
		assert isinstance(n_out, int)
		assert n_out > 0
		assert isinstance(activation, activations.Activation)

		super().__init__(name, trainable)
		self.n_in  = 1
		self.n_out = n_out
		self.weights_init = weights_init
		self.biases_init = biases_init
		self.activation = activation

		self.input_shape = None
		self.output_shape = None

		self.weights = None
		self.biases = None
		self.x = None
		self.z = None

	def add_input_shape_to_layers(self, n_in):
		assert len(n_in) == 1

		self.n_in = n_in[0]
		self.weights = self.weights_init((self.n_in, self.n_out))
		self.biases = self.biases_init((1, self.n_out))

		self.input_shape = '(None,{:4d})'.format(self.n_in)
		self.output_shape = '(None,{:4d})'.format(self.n_out)

		return (self.n_out,)

	def __call__(self, x, training=False):
		assert x.shape[1:] == (self.n_in,)
		self.x = x
		self.z = np.dot(x, self.weights) + self.biases
		return self.activation(self.z)

	def diff(self, prev_da):
		assert self.x is not None

		dz = prev_da * self.activation.diff(self.z)
		m = self.x.shape[-1]

		dw = np.dot(self.x.T, dz) / m
		db = np.sum(dz, axis=0, keepdims=True) / m
		dx = np.dot(dz, self.weights.T,)

		self.x = None
		self.z = None
		return dx, dw, db

	def count_params(self):
		return self.n_out * (self.n_in + 1)

	def get_weights(self):
		return self.weights, self.biases

	def set_weights(self, weights, biases):
		assert weights.shape == (self.n_in, self.n_out)
		assert biases.shape == (1, self.n_out)

		self.weights = np.array(weights)
		self.biases = np.array(biases)


class BatchNorm(Layer):
	__name__ = 'BatchNorm'
	#TODO: Write BatchNorm Layer
	pass


class Dropout(Layer):
	__name__ = 'Dropout'

	def __init__(self, prob, name):
		assert 0 < prob <= 1
		self.rate = prob

		super().__init__(name, False)
		self.n = None
		self.input_shape = None
		self.output_shape = None
		self.filters = None

	def add_input_shape_to_layers(self, n):
		self.n = n
		self.input_shape = self.output_shape = '(None' + (',{:4}'*len(n)).format(*n) + ')'
		return n

	def __call__(self, x, training=False):
		assert x.shape[1:] == self.n
		self.filter = np.random.rand(*x.shape) < self.rate if training else 1
		return self.filter * x

	def diff(self, da):
		dx = self.filter * da
		return dx, 0, 0

	def count_params(self):
		return 0

	def get_weights(self):
		return 0, 0


# CNN Layers:


class Conv1D(Layer):
	__name__ = 'Conv1D'
	pass


class Conv2D(Layer):
	__name__ = 'Conv2D'
	#TODO: Write Conv2D Layer
	pass


class AvgPool(Layer):
	__name__ = 'AvgPool'
	#TODO: Write AvgPool Layer
	pass


class MaxPool(Layer):
	__name__ = 'MaxPool'
	#TODO: Write MaxPool Layer
	pass


class GlobalAvgPool(Layer):
	__name__ = 'GlobalAvgPool'
	pass


class GlobalMaxPool(Layer):
	__name__ = 'GlobalMaxPool'
	pass


class Flatten(Layer):
	__name__ = 'Flatten'
	#TODO: Write Flatten Layer
	pass


class Upsample1D(Layer):
	__name__ = 'Upsample1D'
	pass


class Upsample2D(Layer):
	__name__ = 'Upsample2D'
	pass


# Other Extra Functionality


class Lambda(Layer):
	__name__ = 'Lambda'
	#TODO: Write Lambda Layer
	pass


class Activation(Layer):
	__name__ = 'Activation'
	#TODO: Write Activation Layer
	pass


class Add(Layer):
	__name__ = 'Add'
	#TODO: Write Add Layer
	pass


class Concat(Layer):
	__name__ = 'Concat'
	#TODO: Write Concat Layer
	pass


class Reshape(Layer):
	__name__ = 'Reshape'

	def __init__(self, n_out, name):
		assert isinstance(n_out, tuple)
		for ch in n_out:
			assert ch > 0

		super().__init__(name, False)
		self.n_out = n_out
		self.n_in = None
		self.input_shape = None
		self.output_shape = None

	def add_input_shape_to_layers(self, n_in):
		self.n_in = n_in
		self.input_shape = '(None' + (',{:4}'*len(self.n_in)).format(*self.n_in) + ')'
		self.output_shape = '(None' + (',{:4}'*len(self.n_out)).format(*self.n_out) + ')'
		return self.n_out

	def __call__(self, x, training=False):
		return np.reshape(x, (-1,)+self.n_out)

	def diff(self, da):
		dx = np.reshape(da, (-1,)+self.n_in)
		return dx, 0, 0

	def count_params(self):
		return 0

	def get_weights(self):
		return 0, 0
