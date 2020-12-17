import numpy as np

from ainshamsflow import activations
from ainshamsflow import initializers
from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError
#TODO: Add More Layers


def get(layer_name):
	layers = [Dense, BatchNorm, Dropout,
			  Conv1D, Conv2D, AvgPool, MaxPool, GlobalAvgPool, GlobalMaxPool, Flatten, Upsample1D, Upsample2D,
			  Activation, Reshape]
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
		self.input_shape = None
		self.output_shape = None

	def add_input_shape_to_layer(self, n_in):
		raise BaseClassError

	def __call__(self, x, training=False):
		raise BaseClassError

	def diff(self, da):
		raise BaseClassError

	def count_params(self):
		return 0

	def get_weights(self):
		return 0, 0

	def set_weights(self, weights, biases):
		return

	def summary(self):
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

		return tuple([self.n_out])

	def __call__(self, x, training=False):
		assert x.shape[1:] == (self.n_in,)
		self.x = x
		self.z = np.dot(x, self.weights) + self.biases
		return self.activation(self.z)

	def diff(self, da):
		assert self.x is not None

		dz = da * self.activation.diff(self.z)
		m = self.x.shape[0]

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

	def __init__(self, name, epsilon=0.001, momentum=0.99,
				 gamma_init=initializers.Constant(1), beta_init=initializers.Constant(0),
				 mu_init=initializers.Constant(0), sig_init=initializers.Constant(1),
				 trainable=True):
		assert isinstance(epsilon, float)
		assert 0 < epsilon < 1
		assert isinstance(momentum, float)
		assert 0 <= momentum < 1
		assert isinstance(gamma_init, initializers.Initializer)
		assert isinstance(beta_init, initializers.Initializer)
		assert isinstance(mu_init, initializers.Initializer)
		assert isinstance(sig_init, initializers.Initializer)

		super().__init__(name, trainable)
		self.epsilon = epsilon
		self.momentum = momentum
		self.gamma_init = gamma_init
		self.beta_init = beta_init
		self.mu_init = mu_init
		self.sig_init = sig_init

		self.n = None
		self.input_shape = None
		self.output_shape = None

		self.mu = None
		self.sig = None
		self.gamma = None
		self.beta = None

		self.x_norm = None

	def add_input_shape_to_layers(self, n):
		self.n = (1,) + n
		self.input_shape = self.output_shape = '(None' + (',{:4}' * len(n)).format(*n) + ')'

		self.gamma = self.gamma_init(self.n)
		self.beta = self.beta_init(self.n)
		self.mu = self.mu_init(self.n)
		self.sig = self.sig_init(self.n)

		return n

	def __call__(self, x, training=False):
		if training:
			sample_mu = np.mean(x, axis=0, keepdims=True)
			sample_sig = np.mean(x - sample_mu, axis=0, keepdims=True)
			self.mu  = self.momentum * self.mu  + (1 - self.momentum) * sample_mu
			self.sig = self.momentum * self.sig + (1 - self.momentum) * sample_sig

		self.x_norm = (x - self.mu) / np.sqrt(self.sig + self.epsilon)

		z = self.gamma * self.x_norm + self.beta

		return z

	def diff(self, da):
		m = da.shape[0]

		dgamma = np.sum(da * self.x_norm, axis=0, keepdims=True)
		dbeta = np.sum(da, axis=0, keepdims=True)

		dx_norm = da * self.gamma
		dx = (
				m * dx_norm - np.sum(dx_norm, axis=0) - self.x_norm * np.sum(dx_norm * self.x_norm)
			 ) / (m * np.sqrt(self.sig + self.epsilon))

		return dx, dgamma, dbeta

	def count_params(self):
		return np.prod(self.n) * 2

	def get_weights(self):
		return self.gamma, self.beta

	def set_weights(self, gamma, beta):
		assert gamma.shape == self.n
		assert beta.shape == self.n

		self.gamma = np.array(gamma)
		self.beta = np.array(beta)


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

	def __init__(self, name):
		super().__init__(name, False)
		self.n_out = None
		self.n_in = None
		self.input_shape = None
		self.output_shape = None

	def add_input_shape_to_layers(self, n_in):
		self.n_in = n_in
		self.n_out = (np.prod(n_in),)
		self.input_shape = '(None' + (',{:4}'*len(self.n_in)).format(*self.n_in) + ')'
		self.output_shape = '(None,{:4})'.format(self.n_out[0])
		return self.n_out

	def __call__(self, x, training=False):
		return np.reshape(x, (-1,)+self.n_out)

	def diff(self, da):
		dx = np.reshape(da, (-1,)+self.n_in)
		return dx, 0, 0


class Upsample1D(Layer):
	__name__ = 'Upsample1D'
	pass


class Upsample2D(Layer):
	__name__ = 'Upsample2D'
	pass


# Other Extra Functionality


class Activation(Layer):
	__name__ = 'Activation'

	def __init__(self, act):
		assert isinstance(act, str) or isinstance(act, activations.Activation)
		if isinstance(act, str):
			self.activation = activations.get(act)
			super().__init__(act, False)
		else:
			self.activation = act
			super().__init__(act.__name__, False)

		self.n_in = None
		self.n_out = None
		self.input_shape = None
		self.output_shape = None
		self.x = None

	def add_input_shape_to_layers(self, n_in):
		self.n_in = self.n_out = n_in
		self.input_shape = self.output_shape = '(None' + (',{:4}' * len(n_in)).format(*n_in) + ')'
		return self.n_out

	def __call__(self, x, training=False):
		self.x = x
		return self.activation(x)

	def diff(self, da):
		dx = da * self.activation.diff(self.x)
		return dx, 0, 0


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
