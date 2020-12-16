import numpy as np

from ainshamsflow.activations import Activation
from ainshamsflow.utils.peras_errors import BaseClassError
#TODO: Add More Layers


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

	def summary(self):
		raise BaseClassError


class Dense(Layer):
	def __init__(self, n_out, activation, trainable, name):
		assert isinstance(n_out, int)
		assert isinstance(activation, Activation)

		super().__init__(trainable, name)
		self.input_shape  = (1, None)
		self.output_shape = (n_out, None)
		self.weights = np.random.rand(n_out, 1)
		self.biases = np.zeros((n_out, 1))
		self.activation = activation

		self.x = None
		self.z = None

	def __add_input_shape_to_layers(self, n_in):
		self.input_shape = (n_in, None)
		self.weights = np.random.rand(self.output_shape[0], n_in)
		return self.output_shape[0]

	def __call__(self, x):
		assert x.shape[0] == self.input_shape[0]
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
		return self.output_shape[0] * (self.input_shape[0] + 1)

	def get_weights(self):
		return self.weights, self.biases

	def set_weights(self, weights, biases):
		assert weights.shape == (self.output_shape[0], self.input_shape[0])
		assert biases.shape == (self.output_shape[0], 1)

		self.weights = np.array(weights)
		self.biases = np.array(biases)

	def summary(self):
		input_shape  = (self.weights.shape[1], None)
		output_shape = (self.weights.shape[0], None)
		layer_name   = 'Dense Layer:'
		return '{:20s} {:13d} {:11} {:11} {}'.format(layer_name, self.count_params(), input_shape, output_shape, self.name)
