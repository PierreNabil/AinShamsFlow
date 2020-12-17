import numpy as np

from ainshamsflow import activations
from ainshamsflow import initializers
from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError
#TODO: Add More Layers


def get(layer_name):
	layers = [Dense]
	for layer in layers:
		if layer.__name__.lower() == layer_name.lower():
			return layer
	else:
		raise NameNotFoundError(layer_name, __name__)


class Layer:
	def __init__(self, trainable, name):
		assert isinstance(trainable, bool)
		assert isinstance(name, str)

		self.trainable = trainable
		self.name = name

	def __call__(self, x):
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
		return '{:20s} | {:13d} | {} | {} | {}'.format(self.__name__ + ' Layer:', self.count_params(),
													   self.input_shape, self.output_shape, self.name)


class Dense(Layer):
	__name__ = 'Dense'

	def __init__(self, n_out, name, activation=activations.Linear(),
				 weights_init=initializers.Normal(), biases_init=initializers.Constant(0),
				 trainable=True):
		assert isinstance(n_out, int)
		assert n_out > 0
		assert isinstance(activation, activations.Activation)

		super().__init__(trainable, name)
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
		self.n_in = n_in
		self.weights = self.weights_init((self.n_out, self.n_in))
		self.biases = self.biases_init((self.n_out, 1))

		self.input_shape = '({:4},{})'.format(self.n_in, None)
		self.output_shape = '({:4},{})'.format(self.n_out, None)

		return self.n_out

	def __call__(self, x):
		assert x.shape[0] == self.n_in
		self.x = x
		self.z = np.dot(self.weights, x) + self.biases
		return self.activation(self.z)

	def diff(self, prev_da):
		assert self.x is not None

		dz = prev_da * self.activation.diff(self.z)
		m = self.x.shape[1]

		dw = np.dot(dz, self.x.T) / m
		db = np.sum(dz, axis=1, keepdims=True) / m
		dx = np.dot(self.weights.T, dz)

		self.x = None
		self.z = None
		return dx, dw, db

	def count_params(self):
		return self.n_out * (self.n_in + 1)

	def get_weights(self):
		return self.weights, self.biases

	def set_weights(self, weights, biases):
		assert weights.shape == (self.n_out, self.n_in)
		assert biases.shape == (self.n_out, 1)

		self.weights = np.array(weights)
		self.biases = np.array(biases)
