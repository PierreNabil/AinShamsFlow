"""Layers Module.

In this Module, we include our Layers such as Dense and Conv Layers.
"""

import numpy as np

from ainshamsflow import activations
from ainshamsflow import initializers
from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError, UnsupportedShapeError
#TODO: Add More Layers


def get(layer_name):
	"""Get any Layer in this Module by name."""

	layers = [Dense, BatchNorm, Dropout,
			  Conv1D, Conv2D, AvgPool, MaxPool, GlobalAvgPool, GlobalMaxPool, Flatten, Upsample1D, Upsample2D,
			  Activation, Reshape]
	for layer in layers:
		if layer.__name__.lower() == layer_name.lower():
			return layer
	else:
		raise NameNotFoundError(layer_name, __name__)


class Layer:
	"""Activation Base Class.

	To create a new Layer, create a class that inherits from this class.
	You then have to add any parameters in your constructor
	(while still calling this class' constructor) and redefine the
	__call__(), diff(), add_input_shape_to_layer(), (Manditory)
	count_params(), get_weights() and set_weights() (Optional)
	methods.
	"""

	def __init__(self, name, trainable):
		"""Initialize the name and trainable parameter of the layer."""

		assert isinstance(trainable, bool)

		self.trainable = trainable
		self.name = str(name)
		self.input_shape = None
		self.output_shape = None

	def add_input_shape_to_layer(self, n_in):
		raise BaseClassError

	def __call__(self, x, training=False):
		raise BaseClassError

	def diff(self, da):
		raise BaseClassError

	def count_params(self):
		"""No Parameters in this layer. Returns 0."""
		return 0

	def get_weights(self):
		"""No Parameters in this layer. Returns 0, 0."""
		return np.array([[0]]), np.array([[0]])

	def set_weights(self, weights, biases):
		"""No Parameters in this layer. Do nothing."""
		return

	def summary(self):
		"""return a summary string of the layer.

		used in model.print_summary()
		"""

		return '{:20s} | {:13d} | {:>21} | {:>21} | {:30}'.format(
			self.__name__ + ' Layer:', self.count_params(), self.input_shape, self.output_shape, self.name
		)


# DNN Layers:


class Dense(Layer):
	"""Dense (Fully Connected) Layer."""

	__name__ = 'Dense'

	def __init__(self, n_out, activation=activations.Linear(),
				 weights_init=initializers.Normal(), biases_init=initializers.Constant(0),
				 trainable=True, name=None):
		assert isinstance(n_out, int)
		assert n_out > 0
		if isinstance(activation, str):
			activation = activations.get(activation)
		assert isinstance(activation, activations.Activation)

		super().__init__(name, trainable)
		self.n_in  = (1,)
		self.n_out = (n_out,)
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

		self.n_in = n_in
		self.weights = self.weights_init((self.n_in[0], self.n_out[0]))
		self.biases = self.biases_init((1, self.n_out[0]))

		self.input_shape = '(None,{:4d})'.format(self.n_in[0])
		self.output_shape = '(None,{:4d})'.format(self.n_out[0])

		return self.n_out

	def __call__(self, x, training=False):
		assert x.shape[1:] == self.n_in
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
		return self.n_out[0] * (self.n_in[0] + 1)

	def get_weights(self):
		return self.weights, self.biases

	def set_weights(self, weights, biases):
		assert weights.shape == (self.n_in[0], self.n_out[0])
		assert biases.shape == (1, self.n_out[0])

		self.weights = np.array(weights)
		self.biases = np.array(biases)


class BatchNorm(Layer):
	"""Batch Normalization Layer."""

	__name__ = 'BatchNorm'

	def __init__(self, epsilon=0.001, momentum=0.99,
				 gamma_init=initializers.Constant(1), beta_init=initializers.Constant(0),
				 mu_init=initializers.Constant(0), sig_init=initializers.Constant(1),
				 trainable=True, name=None):
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

		self.n_out = None
		self.input_shape = None
		self.output_shape = None

		self.mu = None
		self.sig = None
		self.gamma = None
		self.beta = None

		self.x_norm = None

	def add_input_shape_to_layers(self, n):
		self.n_out = (1,) + n
		self.input_shape = self.output_shape = '(None' + (',{:4}' * len(n)).format(*n) + ')'

		self.gamma = self.gamma_init(self.n_out)
		self.beta = self.beta_init(self.n_out)
		self.mu = self.mu_init(self.n_out)
		self.sig = self.sig_init(self.n_out)

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
		return np.prod(self.n_out) * 2

	def get_weights(self):
		return self.gamma, self.beta

	def set_weights(self, gamma, beta):
		assert gamma.shape == self.n_out
		assert beta.shape == self.n_out

		self.gamma = np.array(gamma)
		self.beta = np.array(beta)


class Dropout(Layer):
	"""Dropout Layer.

	Remove some neurons from the preveous layer at random
	with probability p.
	"""

	__name__ = 'Dropout'

	def __init__(self, prob, name=None):
		assert 0 < prob <= 1
		self.rate = prob

		super().__init__(name, False)
		self.n_out = None
		self.input_shape = None
		self.output_shape = None
		self.filters = None

	def add_input_shape_to_layers(self, n):
		self.n_out = n
		self.input_shape = self.output_shape = '(None' + (',{:4}'*len(n)).format(*n) + ')'
		return n

	def __call__(self, x, training=False):
		assert x.shape[1:] == self.n_out
		self.filter = np.random.rand(*x.shape) < self.rate if training else 1
		return self.filter * x

	def diff(self, da):
		dx = self.filter * da
		return dx, np.array([[0]]), np.array([[0]])


# CNN Layers:


class Conv1D(Layer):
	"""1-Dimensional Convolution Layer."""

	__name__ = 'Conv1D'
	pass


class Conv2D(Layer):
	"""2-Dimensional Convolution Layer."""

	__name__ = 'Conv2D'
	#TODO: Write Conv2D Layer
	pass


class AvgPool(Layer):
	"""Average Pooling Layer."""

	__name__ = 'AvgPool'
	#TODO: Write AvgPool Layer
	pass


class MaxPool(Layer):
	"""Maximum Pooling Layer."""

	__name__ = 'MaxPool'
	#TODO: Write MaxPool Layer
	pass


class GlobalAvgPool(Layer):
	"""Global Average Pooling Layer."""

	__name__ = 'GlobalAvgPool'
	pass


class GlobalMaxPool(Layer):
	"""Global Maximum Pooling Layer."""

	__name__ = 'GlobalMaxPool'
	pass


class Flatten(Layer):
	"""Flatten Layer.

	Flatten the output of the previous layer into a
	single feature vector.

	Equivalent to Reshape((-1,))
	"""

	__name__ = 'Flatten'

	def __init__(self, name=None):
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
		return dx, np.array([[0]]), np.array([[0]])


class Upsample1D(Layer):
	"""1-Dimensional Up Sampling Layer."""

	__name__ = 'Upsample1D'
	pass


class Upsample2D(Layer):
	"""2-Dimensional Up Sampling Layer."""

	__name__ = 'Upsample2D'
	pass


# Other Extra Functionality


class Activation(Layer):
	"""Activation Layer."""

	__name__ = 'Activation'

	def __init__(self, act, name=None):
		assert isinstance(act, str) or isinstance(act, activations.Activation)
		if isinstance(act, str):
			self.activation = activations.get(act)
		else:
			self.activation = act
			act = act.__name__
		if name is None:
			super().__init__(act, False)
		else:
			super().__init__(name, False)

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
		return dx, np.array([[0]]), np.array([[0]])


class Reshape(Layer):
	"""Reshape Layer.

	Reshape the precious layer's output to any compatible shape.
	"""

	__name__ = 'Reshape'

	def __init__(self, n_out, name=None):
		assert isinstance(n_out, tuple)

		num_of_unk_ch = 0
		self.unk_ch_id = None
		for i, ch in enumerate(n_out):
			if ch == -1 or ch is None:
				if num_of_unk_ch:
					raise UnsupportedShapeError(n_out, 'a shape with less than one unknown dimension.')
				num_of_unk_ch += 1
				self.unk_ch_id = i
			else:
				assert ch > 0

		super().__init__(name, False)
		self.n_out = n_out
		self.n_in = None
		self.input_shape = None
		self.output_shape = None

	def add_input_shape_to_layers(self, n_in):
		self.n_in = n_in

		if self.unk_ch_id is not None:
			n_out = list(self.n_out)
			n_out.pop(self.unk_ch_id)
			new_dim = np.prod(n_in) // np.prod(n_out)
			self.n_out = self.n_out[:self.unk_ch_id] + (new_dim,) + self.n_out[self.unk_ch_id+1:]

		self.input_shape = '(None' + (',{:4}'*len(self.n_in)).format(*self.n_in) + ')'
		self.output_shape = '(None' + (',{:4}'*len(self.n_out)).format(*self.n_out) + ')'
		return self.n_out

	def __call__(self, x, training=False):
		return np.reshape(x, (-1,)+self.n_out)

	def diff(self, da):
		dx = np.reshape(da, (-1,)+self.n_in)
		return dx, np.array([[0]]), np.array([[0]])
