"""Layers Module.

In this Module, we include our Layers such as Dense and Conv Layers.
"""

import numpy as np

from ainshamsflow import activations
from ainshamsflow import initializers
from ainshamsflow.utils.asf_errors import (BaseClassError, NameNotFoundError, UnsupportedShapeError,
										   InvalidShapeError, WrongObjectError, InvalidPreceedingLayerError,
										   InvalidRangeError)


__pdoc__ = dict()

__pdoc__['Layer.add_input_shape_to_layer'] = False
__pdoc__['Layer.diff'] = False
__pdoc__['Layer.count_params'] = False
__pdoc__['Layer.get_weights'] = False
__pdoc__['Layer.set_weights'] = False

for layer_n in ['Dense', 'BatchNorm', 'Dropout',
				'Conv1D', 'Pool1D', 'GlobalPool1D', 'Upsample1D',
				'Conv2D', 'Pool2D', 'GlobalPool2D', 'Upsample2D',
				'Conv3D', 'Pool3D', 'GlobalPool3D', 'Upsample3D',
				'Flatten', 'Activation', 'Reshape']:
	__pdoc__[layer_n + '.__call__'] = True
	__pdoc__[layer_n + '.diff'] = True
	__pdoc__[layer_n + '.add_input_shape_to_layers'] = False


def get(layer_name):
	"""Get any Layer in this Module by name."""

	layers = [Dense, BatchNorm, Dropout,
			  Conv1D, Pool1D, GlobalPool1D, Upsample1D,
			  Conv2D, Pool2D, GlobalPool2D, Upsample2D,
			  Conv3D, Pool3D, GlobalPool3D, Upsample3D,
			  Flatten, Activation, Reshape]
	for layer in layers:
		if layer.__name__.lower() == layer_name.lower():
			return layer
	else:
		raise NameNotFoundError(layer_name, __name__)


class Layer:
	"""Activation Base Class.

	To create a new Layer, create a class that inherits from this class.
	You then have to add any parameters in your constructor
	(while still calling this class' constructor) and redefine the \_\_call\_\_(),
	diff(), add_input_shape_to_layer(), (Manditory)
	count_params(), get_weights() and set_weights() (Optional)
	methods.
	"""

	def __init__(self, name=None, trainable=False):
		"""
		Args:
			name: name of  the layer.
			trainable: Boolean to define whether this layer is trainable or not.
		"""

		if not isinstance(trainable, bool):
			raise WrongObjectError(trainable, True)

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
		"""
		Activations and Initializers can be strings or objects.
		Args:
			n_out: number of (output) neurons in this layer.
			activation: activation function used for the layer.
			weights_init: Initializer for the weights of this layer.
			biases_init: Initializer for the biases of this layers.
			name: name of  the layer.
			trainable: Boolean to define whether this layer is trainable or not.
		"""
		if not isinstance(n_out, int):
			raise WrongObjectError(n_out, 1)
		if n_out <= 0:
			raise InvalidShapeError((n_out,))
		if isinstance(activation, str):
			activation = activations.get(activation)
		if not isinstance(activation, activations.Activation):
			raise WrongObjectError(activation, activations.Activation())
		if not isinstance(weights_init, initializers.Initializer):
			raise WrongObjectError(weights_init, initializers.Initializer())
		if not isinstance(biases_init, initializers.Initializer):
			raise WrongObjectError(biases_init, initializers.Initializer())

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
		if len(n_in) != 1:
			raise InvalidPreceedingLayerError(self)

		self.n_in = n_in
		self.weights = self.weights_init((self.n_in[0], self.n_out[0]))
		self.biases = self.biases_init((1, self.n_out[0]))

		self.input_shape = '(None,{:4d})'.format(self.n_in[0])
		self.output_shape = '(None,{:4d})'.format(self.n_out[0])

		return self.n_out

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)
		self.x = x
		self.z = np.dot(x, self.weights) + self.biases
		return self.activation(self.z)

	def diff(self, da):
		dz = da * self.activation.diff(self.z)
		m = self.x.shape[0]

		dw = np.dot(self.x.T, dz) / m
		db = np.sum(dz, axis=0, keepdims=True) / m
		dx = np.dot(dz, self.weights.T,)

		return dx, dw, db

	def count_params(self):
		return self.n_out[0] * (self.n_in[0] + 1)

	def get_weights(self):
		return self.weights, self.biases

	def set_weights(self, weights, biases):
		if weights.shape != (self.n_in[0], self.n_out[0]):
			raise UnsupportedShapeError(weights.shape, (self.n_in[0], self.n_out[0]))
		if biases.shape != (1, self.n_out[0]):
			raise UnsupportedShapeError(biases.shape, (1, self.n_out[0]))

		self.weights = np.array(weights)
		self.biases = np.array(biases)


class BatchNorm(Layer):
	"""Batch Normalization Layer."""

	__name__ = 'BatchNorm'

	def __init__(self, epsilon=0.001, momentum=0.99,
				 gamma_init=initializers.Constant(1), beta_init=initializers.Constant(0),
				 mu_init=initializers.Constant(0), sig_init=initializers.Constant(1),
				 trainable=True, name=None):
		"""
		Args:
			epsilon:
			momentum:
			gamma_init: Initializer for the weights of this layer.
			beta_init: Initializer for the biases of this layer.
			mu_init: Initializer for the means of this layer.
			sig_init: Initializer for the standard deviations of this layer.
			name: name of  the layer.
			trainable: Boolean to define whether this layer is trainable or not.
		"""
		if not isinstance(epsilon, float):
			raise WrongObjectError(epsilon, float())
		if not 0 < epsilon < 1:
			raise InvalidRangeError(epsilon, 0, 1)
		if not isinstance(momentum, float):
			raise WrongObjectError(momentum, float())
		if not 0 <= momentum < 1:
			raise InvalidRangeError(momentum, 0, 1)
		if not isinstance(gamma_init, initializers.Initializer):
			raise WrongObjectError(gamma_init, initializers.Initializer)
		if not isinstance(beta_init, initializers.Initializer):
			raise WrongObjectError(beta_init, initializers.Initializer)
		if not isinstance(mu_init, initializers.Initializer):
			raise WrongObjectError(mu_init, initializers.Initializer)
		if not isinstance(sig_init, initializers.Initializer):
			raise WrongObjectError(sig_init, initializers.Initializer)

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
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

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
		if gamma.shape != self.n_out:
			raise UnsupportedShapeError(gamma.shape, self.n_out)
		if beta.shape != self.n_out:
			raise UnsupportedShapeError(beta.shape, self.n_out)

		self.gamma = np.array(gamma)
		self.beta = np.array(beta)


class Dropout(Layer):
	"""Dropout Layer.

	Remove some neurons from the preveous layer at random
	with probability p.
	"""

	__name__ = 'Dropout'

	def __init__(self, prob, name=None):
		"""
		Args:
			 prob: probability of keeping a neuron.
			 name: name of  the layer.
		"""
		if not 0 < prob <= 1:
			raise InvalidRangeError(prob, 0, 1)
		self.rate = prob

		super().__init__(name, False)
		self.n_in = None
		self.n_out = None
		self.input_shape = None
		self.output_shape = None
		self.filters = None

	def add_input_shape_to_layers(self, n):
		self.n_out = self.n_in = n
		self.input_shape = self.output_shape = '(None' + (',{:4}'*len(n)).format(*n) + ')'
		return n

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		self.filter = np.random.rand(*x.shape) < self.rate if training else 1
		return self.filter * x

	def diff(self, da):
		dx = self.filter * da
		return dx, np.array([[0]]), np.array([[0]])


# CNN Layers:

class Conv1D(Layer):
	"""1-Dimensional Convolution Layer."""

	__name__ = 'Conv1D'

	def __init__(self, filters, kernel_size, strides=1, padding='valid',
				 kernel_init=initializers.Normal(), biases_init=initializers.Constant(0),
				 activation=activations.Linear(), name=None, trainable=True):
		"""
		Args:
			filters: number of filters (kernels)
			kernel_size: An integer or tuple of 1 integer, specifying the height and
				width of the 1D convolution window. Can be a single integer to specify
				the same value for all spatial dimensions.
			strides: An integer or tuple of 1 integer, specifying the strides of the
				convolution along the height and width. Can be a single integer to
				specify the same value for all spatial dimensions. Specifying any stride
				value != 1 is incompatible with specifying any dilation_rate value != 1.
			padding: one of "valid" or "same" (case-insensitive). "valid" means no padding.
				"same" results in padding evenly to the left/right or up/down of the input such
				that output has the same height/width dimension as the input.
			kernel_init: Initializer for the weights of this layer.
			biases_init: Initializer for the biases of this layer.
			activation: activation function used for the layer.
			name: name of  the layer.
			trainable: Boolean to define whether this layer is trainable or not.
		"""
		if not isinstance(filters, int):
			raise WrongObjectError(filters, 0)
		if isinstance(kernel_size, int):
			kernel_size = (kernel_size,)
		if not isinstance(kernel_size, tuple):
			raise WrongObjectError(kernel_size, tuple())
		if len(kernel_size) != 1:
			raise InvalidShapeError(kernel_size)
		for ch in kernel_size:
			if ch <= 0 or ch % 2 != 1:
				raise InvalidRangeError(kernel_size)
		if isinstance(strides, int):
			strides = (strides,)
		if not isinstance(strides, tuple):
			raise WrongObjectError(strides, tuple())
		if len(strides) != 1:
			raise InvalidShapeError(strides)
		for ch in strides:
			if ch <= 0:
				raise InvalidShapeError(strides)
		padding = padding.lower()
		if not (padding == 'valid' or padding == 'same'):
			raise InvalidRangeError(padding, 'valid', 'same')
		if not isinstance(kernel_init, initializers.Initializer):
			raise WrongObjectError(kernel_init, initializers.Initializer())
		if not isinstance(biases_init, initializers.Initializer):
			raise WrongObjectError(biases_init, initializers.Initializer())
		if isinstance(activation, str):
			activation = activations.get(activation)
		if not isinstance(activation, activations.Activation):
			raise WrongObjectError(activation, activations.Activation())

		super().__init__(name, trainable)
		self.n_in = (None, 1)
		self.n_out = (None, filters)
		self.kernel_init = kernel_init
		self.biases_init = biases_init
		self.activation = activation
		self.strides = strides
		self.same_padding = (padding == 'same')

		self.input_shape = None
		self.output_shape = None

		self.kernel = kernel_size
		self.biases = None

	def add_input_shape_to_layers(self, n_in):
		if len(n_in) != 2:
			raise InvalidPreceedingLayerError(self)

		p_h = 0
		if self.same_padding:
			p_h = (self.kernel[0] - 1) // 2

		h_size = (n_in[0] - self.kernel[0] + 2 * p_h) // self.strides[0] + 1

		self.n_in = n_in
		self.n_out = (h_size, self.n_out[-1])

		self.kernel = self.kernel_init((self.n_out[-1], *self.kernel, self.n_in[-1]))
		self.biases = self.biases_init((self.n_out[-1], 1, 1))

		self.input_shape = '(None,{:4d},{:4d})'.format(*self.n_in)
		self.output_shape = '(None,{:4d},{:4d})'.format(*self.n_out)

		return self.n_out

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		n_c_out, f_h, n_c_in = self.kernel.shape
		m, H_prev, n_c_in = x.shape
		self.x = x

		p_h = 0
		s_h = self.strides[0]
		if self.same_padding:
			p_h = (f_h - 1)//2
			x = np.pad(x, ((0, 0), (p_h, p_h), (0, 0)), mode='constant', constant_values=(0, 0))


		H = int((H_prev - f_h + 2 * p_h) / s_h) + 1

		z = np.zeros((m, H, n_c_out))

		for h in range(H):
			vert_start = s_h * h
			vert_end = s_h * h + f_h
			for c in range(n_c_out):
				z[:, h, c] = (
					np.sum(x[:, vert_start:vert_end, :] * self.kernel[c] + self.biases[c], axis=(1, 2))
				)
		self.z = z
		return self.activation(z)

	def diff(self, da):
		n_c_out, f_h, n_c_in = self.kernel.shape
		m, H, n_c_out = da.shape

		s_h = self.strides[0]
		p_h = 0
		if self.same_padding:
			p_h = (f_h - 1)//2

		dz = da * self.activation.diff(self.z)

		x_pad = np.pad(self.x, ((0, 0), (p_h, p_h), (0, 0)), mode='constant', constant_values=(0, 0))
		dx_pad = np.pad(np.zeros(self.x.shape), ((0, 0), (p_h, p_h), (0, 0)),
						mode='constant', constant_values=(0, 0))
		dw = np.zeros(self.kernel.shape)
		db = np.zeros(self.biases.shape)

		for h in range(H):
			vert_start = s_h * h
			vert_end = s_h * h + f_h
			for c in range(n_c_out):
				dx_pad[:, vert_start:vert_end, :] += (
						self.kernel[c][None, ...] * dz[:, h, c][:, None, None]
				)
				dw[c] += np.sum(
					x_pad[:, vert_start:vert_end, :] * dz[:, h, c][:, None, None],
					axis=0
				)
				db[c] += np.sum(dz[:, h, c], axis=0)

		dx = dx_pad[:, p_h:-p_h, :]
		return dx, dw, db

	def count_params(self):
		return np.prod(self.kernel.shape) + self.n_out[-1]

	def get_weights(self):
		return self.kernel, self.biases

	def set_weights(self, weights, biases):
		if weights.shape != self.kernel.shape:
			raise UnsupportedShapeError(weights.shape, self.kernel.shape)
		if biases.shape != self.biases.shape:
			raise UnsupportedShapeError(biases.shape, self.biases.shape)

		self.kernel = np.array(weights)
		self.biases = np.array(biases)


class Pool1D(Layer):
	"""1-Dimensional Pooling Layer."""

	__name__ = 'Pool1D'

	def __init__(self, pool_size=2, strides=None, padding='valid', mode='max', name=None, trainable=True):
		"""
		Args:
			pool_size: Integer or tuple of 1 integer. Factor by which to downscale. (2,) will halve
			the input dimension. If only one integer is specified, the same window length will be used
			for all dimensions.
			strides: Integer, or tuple of 1 integer. Factor by which to downscale. E.g. 2 will halve the input.
				If None, it will default to pool_size.
			padding: One of "valid" or "same" (case-insensitive). "valid" means no padding. "same"
				results in padding evenly to the left/right or up/down of the input such that output
				has the same height/width dimension as the input.
			mode: One of "max" or "avg" (case-insentitive).
			name: name of  the layer.
			trainable: Boolean to define whether this layer is trainable or not.
		"""
		if isinstance(pool_size, int):
			pool_size = (pool_size,)
		if not isinstance(pool_size, tuple):
			raise WrongObjectError(pool_size, tuple())
		if len(pool_size) != 1:
			raise InvalidShapeError(pool_size)
		for ch in pool_size:
			if ch <= 0:
				raise InvalidShapeError(pool_size)
		if strides is None:
			strides = pool_size
		elif isinstance(strides, int):
			strides = (strides,)
		if not isinstance(strides, tuple):
			raise WrongObjectError(strides, tuple())
		if len(strides) != 1:
			raise InvalidShapeError(strides)
		for ch in strides:
			if ch <= 0:
				raise InvalidShapeError(strides)
		padding = padding.lower()
		if not (padding == 'valid' or padding == 'same'):
			raise InvalidRangeError(padding, 'valid', 'same')
		mode = mode.lower()
		if not (mode == 'max' or mode == 'avg' or mode == 'average' or mode == 'mean'):
			raise InvalidRangeError(mode, 'max', 'avg')
		if mode == 'max':
			self.mode = 'max'
		else:
			self.mode = 'avg'

		super().__init__(name, trainable)
		self.pool_size = pool_size
		self.strides = strides
		self.same_padding = padding == 'same'

		self.n_in = None
		self.n_out = None
		self.input_shape = None
		self.output_shape = None

		self.x = None
		self.z = None

	def add_input_shape_to_layers(self, n_in):
		if len(n_in) != 2:
			raise InvalidPreceedingLayerError(self)

		p_h = 0
		if self.same_padding:
			p_h = (self.pool_size[0] - 1) // 2

		h_size = (n_in[0] - self.pool_size[0] + 2 * p_h) // self.strides[0] + 1

		self.n_in = n_in
		self.n_out = (h_size, n_in[-1])

		self.input_shape = '(None,{:4d},{:4d})'.format(*self.n_in)
		self.output_shape = '(None,{:4d},{:4d})'.format(*self.n_out)

		return self.n_out

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		f_h = self.pool_size[0]
		m, H_prev, n_c = x.shape
		self.x = x

		p_h = 0
		s_h = self.strides[0]
		if self.same_padding:
			p_h = (f_h - 1) // 2
			x = np.pad(x, ((0, 0), (p_h, p_h), (0, 0)), mode='constant', constant_values=(0, 0))

		H = int((H_prev - f_h + 2 * p_h) / s_h) + 1

		z = np.zeros((m, H, n_c))

		for h in range(H):
			vert_start = s_h * h
			vert_end = s_h * h + f_h
			for c in range(n_c):
				if self.mode == 'max':
					func = np.max
				else:
					func = np.mean
				z[:, h, c] = func(x[:, vert_start:vert_end, c], axis=1)

		return z

	def diff(self, da):
		f_h = self.pool_size[0]
		m, H, n_c_out = da.shape

		s_h = self.strides[0]

		dx = np.zeros(self.x.shape)

		for h in range(H):
			vert_start = s_h * h
			vert_end = s_h * h + f_h
			for c in range(n_c_out):
				if self.mode == 'max':
					x_slice = self.x[:, vert_start:vert_end, c]
					mask = np.equal(x_slice, np.max(x_slice, axis=1, keepdims=True))
					dx[:, vert_start:vert_end, c] += (
							mask * np.reshape(da[:, h, c], (-1, 1))
					)

				else:
					da_mean = np.mean(da[:, vert_start:vert_end, c], axis=1)
					dx[:, vert_start:vert_end, c] += (
						da_mean[:, None] / np.prod(self.pool_size) * np.ones(self.pool_size)[None, ...]
					)

		return dx, np.array([[0]]), np.array([[0]])


class GlobalPool1D(Layer):
	"""Global Pooling Layer."""

	__name__ = 'GlobalPool1D'

	def __init__(self, mode='max', name=None):
		"""
		Args:
			mode: One of "max" or "avg" (case-insentitive).
			name: name of  the layer.
		"""
		if not (mode == 'max' or mode == 'avg' or mode == 'average' or mode == 'mean'):
			raise InvalidRangeError(mode, 'max', 'avg')
		if mode == 'max':
			self.mode = 'max'
		else:
			self.mode = 'avg'

		super().__init__(name, False)

		self.n_in = None
		self.n_out = None
		self.input_shape = None
		self.output_shape = None

	def add_input_shape_to_layer(self, n_in):
		if len(n_in) != 2:
			raise InvalidPreceedingLayerError(self)

		self.n_in = n_in
		self.n_out = (n_in[-1],)

		self.input_shape = '(None,{:4d},{:4d})'.format(*self.n_in)
		self.output_shape = '(None,{:4d})'.format(self.n_out[0])

		return self.n_out

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		self.x = x
		if self.mode == 'max':
			self.z = np.max(x, axis=1)
		else:
			self.z = np.mean(x, axis=1)

		return self.z

	def diff(self, da):
		m, H, n_c_out = self.x.shape

		if self.mode == 'max':
			mask = np.equal(self.x, np.max(self.x, axis=1, keepdims=True))
			dx = mask * da.reshape((m, 1, n_c_out))
		else:
			dx = da.reshape((m, 1, n_c_out)).repeat(H, axis=1)

		return dx, np.array([[0]]), np.array([[0]])


class Upsample1D(Layer):
	"""1-Dimensional Up Sampling Layer."""

	__name__ = 'Upsample1D'

	def __init__(self, size=2, name=None):
		"""
		Args:
			size: Int, or tuple of 1 integer. The upsampling factors for rows and columns.
			name: name of  the layer.
		"""
		if isinstance(size, int):
			size = (size,)
		if not isinstance(size, tuple):
			raise WrongObjectError(size, tuple())
		if len(size) != 1:
			raise InvalidShapeError(size)
		for ch in size:
			if ch < 0:
				raise InvalidShapeError(size)

		super().__init__(name, False)
		self.up_size = size

		self.n_in = None
		self.n_out = None
		self.input_shape = None
		self.output_shape = None

	def add_input_shape_to_layers(self, n_in):
		if len(n_in) != 2:
			raise InvalidPreceedingLayerError(self)

		h_size = n_in[0] * self.up_size[0]

		self.n_in = n_in
		self.n_out = (h_size, n_in[-1])

		self.input_shape = '(None,{:4d},{:4d})'.format(*self.n_in)
		self.output_shape = '(None,{:4d},{:4d})'.format(*self.n_out)

		return self.n_out

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		z = x.repeat(self.up_size[0], axis=1)

		return z

	def diff(self, da):
		m, H, n_c_out = da.shape

		tensor_shape = (
			m,
			H // self.up_size[0],
			self.up_size[0],
			n_c_out
		)
		dx = np.reshape(da, tensor_shape).sum(axis=2)

		return dx, np.array([[0]]), np.array([[0]])


class Conv2D(Layer):
	"""2-Dimensional Convolution Layer."""

	__name__ = 'Conv2D'

	def __init__(self, filters, kernel_size, strides=1, padding='valid',
				 kernel_init=initializers.Normal(), biases_init=initializers.Constant(0),
				 activation=activations.Linear(), name=None, trainable=True):
		"""
		Args:
			filters: number of filters (kernels)
			kernel_size: An integer or tuple of 2 integers, specifying the height and
				width of the 2D convolution window. Can be a single integer to specify
				the same value for all spatial dimensions.
			strides: An integer or tuple of 2 integers, specifying the strides of the
				convolution along the height and width. Can be a single integer to
				specify the same value for all spatial dimensions. Specifying any stride
				value != 1 is incompatible with specifying any dilation_rate value != 1.
			padding: one of "valid" or "same" (case-insensitive). "valid" means no padding.
				"same" results in padding evenly to the left/right or up/down of the input such
				that output has the same height/width dimension as the input.
			kernel_init: Initializer for the weights of this layer.
			biases_init: Initializer for the biases of this layer.
			activation: activation function used for the layer.
			name: name of  the layer.
			trainable: Boolean to define whether this layer is trainable or not.
		"""
		if not isinstance(filters, int):
			raise WrongObjectError(filters, 0)
		if isinstance(kernel_size, int):
			kernel_size = (kernel_size, kernel_size)
		if not isinstance(kernel_size, tuple):
			raise WrongObjectError(kernel_size, tuple())
		if len(kernel_size) != 2:
			raise InvalidShapeError(kernel_size)
		for ch in kernel_size:
			if ch <= 0 or ch % 2 != 1:
				raise InvalidRangeError(kernel_size)
		if isinstance(strides, int):
			strides = (strides, strides)
		if not isinstance(strides, tuple):
			raise WrongObjectError(strides, tuple())
		if len(strides) != 2:
			raise InvalidShapeError(strides)
		for ch in strides:
			if ch <= 0:
				raise InvalidShapeError(strides)
		padding = padding.lower()
		if not (padding == 'valid' or padding == 'same'):
			raise InvalidRangeError(padding, 'valid', 'same')
		if not isinstance(kernel_init, initializers.Initializer):
			raise WrongObjectError(kernel_init, initializers.Initializer())
		if not isinstance(biases_init, initializers.Initializer):
			raise WrongObjectError(biases_init, initializers.Initializer())
		if isinstance(activation, str):
			activation = activations.get(activation)
		if not isinstance(activation, activations.Activation):
			raise WrongObjectError(activation, activations.Activation())

		super().__init__(name, trainable)
		self.n_in = (None, None, 1)
		self.n_out = (None, None, filters)
		self.kernel_init = kernel_init
		self.biases_init = biases_init
		self.activation = activation
		self.strides = strides
		self.same_padding = (padding == 'same')

		self.input_shape = None
		self.output_shape = None

		self.kernel = kernel_size
		self.biases = None

	def add_input_shape_to_layers(self, n_in):
		if len(n_in) != 3:
			raise InvalidPreceedingLayerError(self)

		p_h, p_w = 0, 0
		if self.same_padding:
			p_h = (self.kernel[0] - 1) // 2
			p_w = (self.kernel[1] - 1) // 2

		h_size = (n_in[0] - self.kernel[0] + 2 * p_h) // self.strides[0] + 1
		w_size = (n_in[1] - self.kernel[1] + 2 * p_w) // self.strides[1] + 1

		self.n_in = n_in
		self.n_out = (h_size, w_size, self.n_out[-1])

		self.kernel = self.kernel_init((self.n_out[-1], *self.kernel, self.n_in[-1]))
		self.biases = self.biases_init((self.n_out[-1], 1, 1, 1))

		self.input_shape = '(None,{:4d},{:4d},{:4d})'.format(*self.n_in)
		self.output_shape = '(None,{:4d},{:4d},{:4d})'.format(*self.n_out)

		return self.n_out

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		n_c_out, f_h, f_w, n_c_in = self.kernel.shape
		m, H_prev, W_prev, n_c_in = x.shape
		self.x = x

		p_h, p_w = 0, 0
		s_h, s_w = self.strides
		if self.same_padding:
			p_h, p_w = (f_h - 1)//2, (f_w - 1)//2
			x = np.pad(x, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)), mode='constant', constant_values=(0, 0))


		H = int((H_prev - f_h + 2 * p_h) / s_h) + 1
		W = int((W_prev - f_w + 2 * p_w) / s_w) + 1

		z = np.zeros((m, H, W, n_c_out))

		for h in range(H):
			vert_start = s_h * h
			vert_end = s_h * h + f_h
			for w in range(W):
				horiz_start = s_w * w
				horiz_end = s_w * w + f_w
				for c in range(n_c_out):
					z[:, h, w, c] = (
						np.sum(x[:, vert_start:vert_end, horiz_start:horiz_end, :] * self.kernel[c] + self.biases[c], axis=(1, 2, 3))
					)
		self.z = z
		return self.activation(z)

	def diff(self, da):
		n_c_out, f_h, f_w, n_c_in = self.kernel.shape
		m, H, W, n_c_out = da.shape

		s_h, s_w = self.strides
		p_h, p_w = 0, 0
		if self.same_padding:
			p_h, p_w = (f_h - 1)//2, (f_w - 1)//2

		dz = da * self.activation.diff(self.z)

		x_pad = np.pad(self.x, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)), mode='constant', constant_values=(0, 0))
		dx_pad = np.pad(np.zeros(self.x.shape), ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)),
						mode='constant', constant_values=(0, 0))
		dw = np.zeros(self.kernel.shape)
		db = np.zeros(self.biases.shape)

		for h in range(H):
			vert_start = s_h * h
			vert_end = s_h * h + f_h
			for w in range(W):
				horiz_start = s_w * w
				horiz_end = s_w * w + f_w
				for c in range(n_c_out):
					dx_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += (
							self.kernel[c][None, ...] * dz[:, h, w, c][:, None, None, None]
					)
					dw[c] += np.sum(
						x_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] * dz[:, h, w, c][:, None, None, None],
						axis=0
					)
					db[c] += np.sum(dz[:, h, w, c], axis=0)

		dx = dx_pad[:, p_h:-p_h, p_w:-p_w, :]
		return dx, dw, db

	def count_params(self):
		return np.prod(self.kernel.shape) + self.n_out[-1]

	def get_weights(self):
		return self.kernel, self.biases

	def set_weights(self, weights, biases):
		if weights.shape != self.kernel.shape:
			raise UnsupportedShapeError(weights.shape, self.kernel.shape)
		if biases.shape != self.biases.shape:
			raise UnsupportedShapeError(biases.shape, self.biases.shape)

		self.kernel = np.array(weights)
		self.biases = np.array(biases)


class Pool2D(Layer):
	"""2-Dimensional Pooling Layer."""

	__name__ = 'Pool2D'

	def __init__(self, pool_size=2, strides=None, padding='valid', mode='max', name=None, trainable=True):
		"""
		Args:
			pool_size: Integer or tuple of 2 integers, factors by which to downscale. (2, 2) will halve
			the input dimensions. If only one integer is specified, the same window length will be used
			for all dimensions.
			strides: Integer, or tuple of 2 integers. Factor by which to downscale. 2 will halve the input.
				If None, it will default to pool_size.
			padding: One of "valid" or "same" (case-insensitive). "valid" means no padding. "same"
				results in padding evenly to the left/right or up/down of the input such that output
				has the same height/width dimension as the input.
			mode: One of "max" or "avg" (case-insentitive).
			name: name of  the layer.
			trainable: Boolean to define whether this layer is trainable or not.
		"""
		if isinstance(pool_size, int):
			pool_size = (pool_size, pool_size)
		if not isinstance(pool_size, tuple):
			raise WrongObjectError(pool_size, tuple())
		if len(pool_size) != 2:
			raise InvalidShapeError(pool_size)
		for ch in pool_size:
			if ch <= 0:
				raise InvalidShapeError(pool_size)
		if strides is None:
			strides = pool_size
		elif isinstance(strides, int):
			strides = (strides, strides)
		if not isinstance(strides, tuple):
			raise WrongObjectError(strides, tuple())
		if len(strides) != 2:
			raise InvalidShapeError(strides)
		for ch in strides:
			if ch <= 0:
				raise InvalidShapeError(strides)
		padding = padding.lower()
		if not (padding == 'valid' or padding == 'same'):
			raise InvalidRangeError(padding, 'valid', 'same')
		mode = mode.lower()
		if not (mode == 'max' or mode == 'avg' or mode == 'average' or mode == 'mean'):
			raise InvalidRangeError(mode, 'max', 'avg')
		if mode == 'max':
			self.mode = 'max'
		else:
			self.mode = 'avg'

		super().__init__(name, trainable)
		self.pool_size = pool_size
		self.strides = strides
		self.same_padding = padding == 'same'

		self.n_in = None
		self.n_out = None
		self.input_shape = None
		self.output_shape = None

		self.x = None
		self.z = None

	def add_input_shape_to_layers(self, n_in):
		if len(n_in) != 3:
			raise InvalidPreceedingLayerError(self)

		p_h, p_w = 0, 0
		if self.same_padding:
			p_h = (self.pool_size[0] - 1) // 2
			p_w = (self.pool_size[1] - 1) // 2

		h_size = (n_in[0] - self.pool_size[0] + 2 * p_h) // self.strides[0] + 1
		w_size = (n_in[1] - self.pool_size[1] + 2 * p_w) // self.strides[1] + 1

		self.n_in = n_in
		self.n_out = (h_size, w_size, n_in[-1])

		self.input_shape = '(None,{:4d},{:4d},{:4d})'.format(*self.n_in)
		self.output_shape = '(None,{:4d},{:4d},{:4d})'.format(*self.n_out)

		return self.n_out

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		f_h, f_w = self.pool_size
		m, H_prev, W_prev, n_c = x.shape
		self.x = x

		p_h, p_w = 0, 0
		s_h, s_w = self.strides
		if self.same_padding:
			p_h, p_w = (f_h - 1) // 2, (f_w - 1) // 2
			x = np.pad(x, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)), mode='constant', constant_values=(0, 0))

		H = int((H_prev - f_h + 2 * p_h) / s_h) + 1
		W = int((W_prev - f_w + 2 * p_w) / s_w) + 1

		z = np.zeros((m, H, W, n_c))

		for h in range(H):
			vert_start = s_h * h
			vert_end = s_h * h + f_h
			for w in range(W):
				horiz_start = s_w * w
				horiz_end = s_w * w + f_w
				for c in range(n_c):
					if self.mode == 'max':
						func = np.max
					else:
						func = np.mean
					z[:, h, w, c] = func(x[:, vert_start:vert_end, horiz_start:horiz_end, c], axis=(1, 2))

		return z

	def diff(self, da):
		f_h, f_w = self.pool_size
		m, H, W, n_c_out = da.shape

		s_h, s_w = self.strides

		dx = np.zeros(self.x.shape)

		for h in range(H):
			vert_start = s_h * h
			vert_end = s_h * h + f_h
			for w in range(W):
				horiz_start = s_w * w
				horiz_end = s_w * w + f_w
				for c in range(n_c_out):
					if self.mode == 'max':
						x_slice = self.x[:, vert_start:vert_end, horiz_start:horiz_end, c]
						mask = np.equal(x_slice, np.max(x_slice, axis=(1, 2), keepdims=True))
						dx[:, vert_start:vert_end, horiz_start:horiz_end, c] += (
								mask * np.reshape(da[:, h, w, c], (-1, 1, 1))
						)

					else:
						da_mean = np.mean(da[:, vert_start:vert_end, horiz_start:horiz_end, c], axis=(1, 2))
						dx[:, vert_start:vert_end, horiz_start:horiz_end, c] += (
							da_mean[:, None, None] / np.prod(self.pool_size) * np.ones(self.pool_size)[None, ...]
						)

		return dx, np.array([[0]]), np.array([[0]])


class GlobalPool2D(Layer):
	"""Global Pooling Layer."""

	__name__ = 'GlobalPool2D'

	def __init__(self, mode='max', name=None):
		"""
		Args:
			mode: One of "max" or "avg" (case-insentitive).
			name: name of  the layer.
		"""
		if not (mode == 'max' or mode == 'avg' or mode == 'average' or mode == 'mean'):
			raise InvalidRangeError(mode, 'max', 'avg')
		if mode == 'max':
			self.mode = 'max'
		else:
			self.mode = 'avg'

		super().__init__(name, False)

		self.n_in = None
		self.n_out = None
		self.input_shape = None
		self.output_shape = None

	def add_input_shape_to_layer(self, n_in):
		if len(n_in) != 3:
			raise InvalidPreceedingLayerError(self)

		self.n_in = n_in
		self.n_out = (n_in[-1],)

		self.input_shape = '(None,{:4d},{:4d},{:4d})'.format(*self.n_in)
		self.output_shape = '(None,{:4d})'.format(self.n_out[0])

		return self.n_out

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		self.x = x
		if self.mode == 'max':
			self.z = np.max(x, axis=(1, 2))
		else:
			self.z = np.mean(x, axis=(1, 2))

		return self.z

	def diff(self, da):
		m, H, W, n_c_out = self.x.shape

		if self.mode == 'max':
			mask = np.equal(self.x, np.max(self.x, axis=(1, 2), keepdims=True))
			dx = mask * da.reshape((m, 1, 1, n_c_out))
		else:
			dx = da.reshape((m, 1, 1, n_c_out)).repeat(H, axis=1).repeat(W, axis=2)

		return dx, np.array([[0]]), np.array([[0]])


class Upsample2D(Layer):
	"""2-Dimensional Up Sampling Layer."""

	__name__ = 'Upsample2D'

	def __init__(self, size=2, name=None):
		"""
		Args:
			size: Int, or tuple of 2 integers. The upsampling factors for rows and columns.
			name: name of  the layer.
		"""
		if isinstance(size, int):
			size = (size, size)
		if not isinstance(size, tuple):
			raise WrongObjectError(size, tuple())
		if len(size) != 2:
			raise InvalidShapeError(size)
		for ch in size:
			if ch < 0:
				raise InvalidShapeError(size)

		super().__init__(name, False)
		self.up_size = size

		self.n_in = None
		self.n_out = None
		self.input_shape = None
		self.output_shape = None

	def add_input_shape_to_layers(self, n_in):
		if len(n_in) != 3:
			raise InvalidPreceedingLayerError(self)

		h_size = n_in[0] * self.up_size[0]
		w_size = n_in[1] * self.up_size[1]

		self.n_in = n_in
		self.n_out = (h_size, w_size, n_in[-1])

		self.input_shape = '(None,{:4d},{:4d},{:4d})'.format(*self.n_in)
		self.output_shape = '(None,{:4d},{:4d},{:4d})'.format(*self.n_out)

		return self.n_out

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		z = x.repeat(self.up_size[0], axis=1).repeat(self.up_size[1], axis=2)

		return z

	def diff(self, da):
		m, H, W, n_c_out = da.shape

		tensor_shape = (
			m,
			H // self.up_size[0],
			self.up_size[0],
			W // self.up_size[1],
			self.up_size[1],
			n_c_out
		)
		dx = np.reshape(da, tensor_shape).sum(axis=(2, 4))

		return dx, np.array([[0]]), np.array([[0]])


class Conv3D(Layer):
	"""3-Dimensional Convolution Layer."""

	__name__ = 'Conv3D'

	def __init__(self, filters, kernel_size, strides=1, padding='valid',
				 kernel_init=initializers.Normal(), biases_init=initializers.Constant(0),
				 activation=activations.Linear(), name=None, trainable=True):
		"""
		Args:
			filters: number of filters (kernels)
			kernel_size: An integer or tuple of 3 integers, specifying the height and
				width of the 3D convolution window. Can be a single integer to specify
				the same value for all spatial dimensions.
			strides: An integer or tuple of 3 integers, specifying the strides of the
				convolution along the height and width. Can be a single integer to
				specify the same value for all spatial dimensions. Specifying any stride
				value != 1 is incompatible with specifying any dilation_rate value != 1.
			padding: one of "valid" or "same" (case-insensitive). "valid" means no padding.
				"same" results in padding evenly to the left/right or up/down of the input such
				that output has the same height/width dimension as the input.
			kernel_init: Initializer for the weights of this layer.
			biases_init: Initializer for the biases of this layer.
			activation: activation function used for the layer.
			name: name of  the layer.
			trainable: Boolean to define whether this layer is trainable or not.
		"""
		if not isinstance(filters, int):
			raise WrongObjectError(filters, 0)
		if isinstance(kernel_size, int):
			kernel_size = (kernel_size, kernel_size, kernel_size)
		if not isinstance(kernel_size, tuple):
			raise WrongObjectError(kernel_size, tuple())
		if len(kernel_size) != 3:
			raise InvalidShapeError(kernel_size)
		for ch in kernel_size:
			if ch <= 0 or ch % 2 != 1:
				raise InvalidRangeError(kernel_size)
		if isinstance(strides, int):
			strides = (strides, strides, strides)
		if not isinstance(strides, tuple):
			raise WrongObjectError(strides, tuple())
		if len(strides) != 3:
			raise InvalidShapeError(strides)
		for ch in strides:
			if ch <= 0:
				raise InvalidShapeError(strides)
		padding = padding.lower()
		if not (padding == 'valid' or padding == 'same'):
			raise InvalidRangeError(padding, 'valid', 'same')
		if not isinstance(kernel_init, initializers.Initializer):
			raise WrongObjectError(kernel_init, initializers.Initializer())
		if not isinstance(biases_init, initializers.Initializer):
			raise WrongObjectError(biases_init, initializers.Initializer())
		if isinstance(activation, str):
			activation = activations.get(activation)
		if not isinstance(activation, activations.Activation):
			raise WrongObjectError(activation, activations.Activation())

		super().__init__(name, trainable)
		self.n_in = (None, None, None, 1)
		self.n_out = (None, None, None, filters)
		self.kernel_init = kernel_init
		self.biases_init = biases_init
		self.activation = activation
		self.strides = strides
		self.same_padding = (padding == 'same')

		self.input_shape = None
		self.output_shape = None

		self.kernel = kernel_size
		self.biases = None

	def add_input_shape_to_layers(self, n_in):
		if len(n_in) != 4:
			raise InvalidPreceedingLayerError(self)

		p_h, p_w, p_d = 0, 0, 0
		if self.same_padding:
			p_h = (self.kernel[0] - 1) // 2
			p_w = (self.kernel[1] - 1) // 2
			p_d = (self.kernel[2] - 1) // 2

		h_size = (n_in[0] - self.kernel[0] + 2 * p_h) // self.strides[0] + 1
		w_size = (n_in[1] - self.kernel[1] + 2 * p_w) // self.strides[1] + 1
		d_size = (n_in[2] - self.kernel[2] + 2 * p_d) // self.strides[2] + 1

		self.n_in = n_in
		self.n_out = (h_size, w_size, d_size, self.n_out[-1])

		self.kernel = self.kernel_init((self.n_out[-1], *self.kernel, self.n_in[-1]))
		self.biases = self.biases_init((self.n_out[-1], 1, 1, 1, 1))

		self.input_shape = '(None,{:4d},{:4d},{:4d},{:4d})'.format(*self.n_in)
		self.output_shape = '(None,{:4d},{:4d},{:4d},{:4d})'.format(*self.n_out)

		return self.n_out

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		n_c_out, f_h, f_w, f_d, n_c_in = self.kernel.shape
		m, H_prev, W_prev, D_prev, n_c_in = x.shape
		self.x = x

		p_h, p_w, p_d = 0, 0, 0
		s_h, s_w, s_d = self.strides
		if self.same_padding:
			p_h, p_w, p_d = (f_h - 1)//2, (f_w - 1)//2, (f_d - 1)//2
			x = np.pad(x, ((0, 0), (p_h, p_h), (p_w, p_w), (p_d, p_d), (0, 0)), mode='constant', constant_values=(0, 0))


		H = int((H_prev - f_h + 2 * p_h) / s_h) + 1
		W = int((W_prev - f_w + 2 * p_w) / s_w) + 1
		D = int((D_prev - f_d + 2 * p_d) / s_d) + 1

		z = np.zeros((m, H, W, D, n_c_out))

		for h in range(H):
			vert_start = s_h * h
			vert_end = s_h * h + f_h
			for w in range(W):
				horiz_start = s_w * w
				horiz_end = s_w * w + f_w
				for d in range(D):
					depth_start = s_d * d
					depth_end = s_d * d + f_d
					for c in range(n_c_out):
						z[:, h, w, d, c] = (
							np.sum(x[:, vert_start:vert_end, horiz_start:horiz_end, depth_start:depth_end, :]
								   * self.kernel[c] + self.biases[c], axis=(1, 2, 3, 4))
						)
		self.z = z
		return self.activation(z)

	def diff(self, da):
		n_c_out, f_h, f_w, f_d, n_c_in = self.kernel.shape
		m, H, W, D, n_c_out = da.shape

		s_h, s_w, s_d = self.strides
		p_h, p_w, p_d = 0, 0, 0
		if self.same_padding:
			p_h, p_w, p_d = (f_h - 1)//2, (f_w - 1)//2, (f_d - 1)//2

		dz = da * self.activation.diff(self.z)

		x_pad = np.pad(self.x, ((0, 0), (p_h, p_h), (p_w, p_w), (p_d, p_d), (0, 0)), mode='constant', constant_values=(0, 0))
		dx_pad = np.zeros(self.x_pad.shape)
		dw = np.zeros(self.kernel.shape)
		db = np.zeros(self.biases.shape)

		for h in range(H):
			vert_start = s_h * h
			vert_end = s_h * h + f_h
			for w in range(W):
				horiz_start = s_w * w
				horiz_end = s_w * w + f_w
				for d in range(D):
					depth_start = s_d * d
					depth_end = s_d * d + f_d
					for c in range(n_c_out):
						dx_pad[:, vert_start:vert_end, horiz_start:horiz_end, depth_start:depth_end, :] += (
								self.kernel[c][None, ...] * dz[:, h, w, c][:, None, None, None, None]
						)
						dw[c] += np.sum(
							x_pad[:, vert_start:vert_end, horiz_start:horiz_end, depth_start:depth_end, :]
							* dz[:, h, w, d, c][:, None, None, None, None],
							axis=0
						)
						db[c] += np.sum(dz[:, h, w, d, c], axis=0)

		dx = dx_pad[:, p_h:-p_h, p_w:-p_w, p_d:-p_d, :]
		return dx, dw, db

	def count_params(self):
		return np.prod(self.kernel.shape) + self.n_out[-1]

	def get_weights(self):
		return self.kernel, self.biases

	def set_weights(self, weights, biases):
		if weights.shape != self.kernel.shape:
			raise UnsupportedShapeError(weights.shape, self.kernel.shape)
		if biases.shape != self.biases.shape:
			raise UnsupportedShapeError(biases.shape, self.biases.shape)

		self.kernel = np.array(weights)
		self.biases = np.array(biases)


class Pool3D(Layer):
	"""3-Dimensional Pooling Layer."""

	__name__ = 'Pool3D'

	def __init__(self, pool_size=2, strides=None, padding='valid', mode='max', name=None, trainable=True):
		"""
		Args:
			pool_size: Integer or tuple of 3 integers, factors by which to downscale. (2, 2, 2) will halve
			the input dimensions. If only one integer is specified, the same window length will be used
			for all dimensions.
			strides: Integer, or tuple of 3 integers. Factor by which to downscale. 2 will halve the input.
				If None, it will default to pool_size.
			padding: One of "valid" or "same" (case-insensitive). "valid" means no padding. "same"
				results in padding evenly to the left/right or up/down of the input such that output
				has the same height/width dimension as the input.
			mode: One of "max" or "avg" (case-insentitive).
			name: name of  the layer.
			trainable: Boolean to define whether this layer is trainable or not.
		"""
		if isinstance(pool_size, int):
			pool_size = (pool_size, pool_size, pool_size)
		if not isinstance(pool_size, tuple):
			raise WrongObjectError(pool_size, tuple())
		if len(pool_size) != 3:
			raise InvalidShapeError(pool_size)
		for ch in pool_size:
			if ch <= 0:
				raise InvalidShapeError(pool_size)
		if strides is None:
			strides = pool_size
		elif isinstance(strides, int):
			strides = (strides, strides, strides)
		if not isinstance(strides, tuple):
			raise WrongObjectError(strides, tuple())
		if len(strides) != 3:
			raise InvalidShapeError(strides)
		for ch in strides:
			if ch <= 0:
				raise InvalidShapeError(strides)
		padding = padding.lower()
		if not (padding == 'valid' or padding == 'same'):
			raise InvalidRangeError(padding, 'valid', 'same')
		mode = mode.lower()
		if not (mode == 'max' or mode == 'avg' or mode == 'average' or mode == 'mean'):
			raise InvalidRangeError(mode, 'max', 'avg')
		if mode == 'max':
			self.mode = 'max'
		else:
			self.mode = 'avg'

		super().__init__(name, trainable)
		self.pool_size = pool_size
		self.strides = strides
		self.same_padding = padding == 'same'

		self.n_in = None
		self.n_out = None
		self.input_shape = None
		self.output_shape = None

		self.x = None
		self.z = None

	def add_input_shape_to_layers(self, n_in):
		if len(n_in) != 4:
			raise InvalidPreceedingLayerError(self)

		p_h, p_w, p_d = 0, 0, 0
		if self.same_padding:
			p_h = (self.pool_size[0] - 1) // 2
			p_w = (self.pool_size[1] - 1) // 2
			p_d = (self.pool_size[2] - 1) // 2

		h_size = (n_in[0] - self.pool_size[0] + 2 * p_h) // self.strides[0] + 1
		w_size = (n_in[1] - self.pool_size[1] + 2 * p_w) // self.strides[1] + 1
		d_size = (n_in[2] - self.pool_size[2] + 2 * p_d) // self.strides[2] + 1

		self.n_in = n_in
		self.n_out = (h_size, w_size, d_size, n_in[-1])

		self.input_shape = '(None,{:4d},{:4d},{:4d},{:4d})'.format(*self.n_in)
		self.output_shape = '(None,{:4d},{:4d},{:4d},{:4d})'.format(*self.n_out)

		return self.n_out

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		f_h, f_w, f_d = self.pool_size
		m, H_prev, W_prev, D_prev, n_c = x.shape
		self.x = x

		p_h, p_w, p_d = 0, 0, 0
		s_h, s_w, s_d = self.strides
		if self.same_padding:
			p_h, p_w, p_d = (f_h - 1) // 2, (f_w - 1) // 2, (f_d - 1) // 2
			x = np.pad(x, ((0, 0), (p_h, p_h), (p_w, p_w), (p_d, p_d), (0, 0)), mode='constant', constant_values=(0, 0))

		H = int((H_prev - f_h + 2 * p_h) / s_h) + 1
		W = int((W_prev - f_w + 2 * p_w) / s_w) + 1
		D = int((D_prev - f_d + 2 * p_d) / s_d) + 1

		z = np.zeros((m, H, W, D, n_c))

		for h in range(H):
			vert_start = s_h * h
			vert_end = s_h * h + f_h
			for w in range(W):
				horiz_start = s_w * w
				horiz_end = s_w * w + f_w
				for d in range(D):
					depth_start = s_d * d
					depth_end = s_d * d + f_d
					for c in range(n_c):
						if self.mode == 'max':
							func = np.max
						else:
							func = np.mean
						z[:, h, w, d, c] = func(x[:, vert_start:vert_end, horiz_start:horiz_end, depth_start:depth_end,
												c], axis=(1, 2, 3))

		return z

	def diff(self, da):
		f_h, f_w, f_d = self.pool_size
		m, H, W, D, n_c_out = da.shape

		s_h, s_w, s_d = self.strides

		dx = np.zeros(self.x.shape)

		for h in range(H):
			vert_start = s_h * h
			vert_end = s_h * h + f_h
			for w in range(W):
				horiz_start = s_w * w
				horiz_end = s_w * w + f_w
				for d in range(D):
					depth_start = s_d * d
					depth_end = s_d * d + f_d
					for c in range(n_c_out):
						if self.mode == 'max':
							x_slice = self.x[:, vert_start:vert_end, horiz_start:horiz_end, depth_start:depth_end, c]
							mask = np.equal(x_slice, np.max(x_slice, axis=(1, 2, 3), keepdims=True))
							dx[:, vert_start:vert_end, horiz_start:horiz_end, depth_start:, depth_end, c] += (
									mask * np.reshape(da[:, h, w, d, c], (-1, 1, 1, 1))
							)

						else:
							da_mean = np.mean(da[:, vert_start:vert_end, horiz_start:horiz_end, depth_start:depth_end,
											  c], axis=(1, 2, 3))
							dx[:, vert_start:vert_end, horiz_start:horiz_end, depth_start:depth_end, c] += (
								da_mean[:, None, None, None] / np.prod(self.pool_size) * np.ones(self.pool_size)[None, ...]
							)

		return dx, np.array([[0]]), np.array([[0]])


class GlobalPool3D(Layer):
	"""Global Pooling Layer."""

	__name__ = 'GlobalPool3D'

	def __init__(self, mode='max', name=None):
		"""
		Args:
			mode: One of "max" or "avg" (case-insentitive).
			name: name of  the layer.
		"""
		if not (mode == 'max' or mode == 'avg' or mode == 'average' or mode == 'mean'):
			raise InvalidRangeError(mode, 'max', 'avg')
		if mode == 'max':
			self.mode = 'max'
		else:
			self.mode = 'avg'

		super().__init__(name, False)

		self.n_in = None
		self.n_out = None
		self.input_shape = None
		self.output_shape = None

	def add_input_shape_to_layer(self, n_in):
		if len(n_in) != 4:
			raise InvalidPreceedingLayerError(self)

		self.n_in = n_in
		self.n_out = (n_in[-1],)

		self.input_shape = '(None,{:4d},{:4d},{:4d},{:4d})'.format(*self.n_in)
		self.output_shape = '(None,{:4d})'.format(self.n_out[0])

		return self.n_out

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		self.x = x
		if self.mode == 'max':
			self.z = np.max(x, axis=(1, 2, 3))
		else:
			self.z = np.mean(x, axis=(1, 2, 3))

		return self.z

	def diff(self, da):
		m, H, W, D, n_c_out = self.x.shape

		if self.mode == 'max':
			mask = np.equal(self.x, np.max(self.x, axis=(1, 2, 3), keepdims=True))
			dx = mask * da.reshape((m, 1, 1, 1, n_c_out))
		else:
			dx = da.reshape((m, 1, 1, 1, n_c_out)).repeat(H, axis=1).repeat(W, axis=2).repeat(D, axis=3)

		return dx, np.array([[0]]), np.array([[0]])


class Upsample3D(Layer):
	"""3-Dimensional Up Sampling Layer."""

	__name__ = 'Upsample3D'

	def __init__(self, size=2, name=None):
		"""
		Args:
			size: Int, or tuple of 2 integers. The upsampling factors for rows and columns.
			name: name of  the layer.
		"""
		if isinstance(size, int):
			size = (size, size, size)
		if not isinstance(size, tuple):
			raise WrongObjectError(size, tuple())
		if len(size) != 3:
			raise InvalidShapeError(size)
		for ch in size:
			if ch < 0:
				raise InvalidShapeError(size)

		super().__init__(name, False)
		self.up_size = size

		self.n_in = None
		self.n_out = None
		self.input_shape = None
		self.output_shape = None

	def add_input_shape_to_layers(self, n_in):
		if len(n_in) != 4:
			raise InvalidPreceedingLayerError(self)

		h_size = n_in[0] * self.up_size[0]
		w_size = n_in[1] * self.up_size[1]
		d_size = n_in[2] * self.up_size[2]

		self.n_in = n_in
		self.n_out = (h_size, w_size, d_size, n_in[-1])

		self.input_shape = '(None,{:4d},{:4d},{:4d},{:4d})'.format(*self.n_in)
		self.output_shape = '(None,{:4d},{:4d},{:4d},{:4d})'.format(*self.n_out)

		return self.n_out

	def __call__(self, x, training=False):
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		z = x.repeat(self.up_size[0], axis=1).repeat(self.up_size[1], axis=2).repeat(self.up_size[2], axis=3)

		return z

	def diff(self, da):
		m, H, W, D, n_c_out = da.shape

		tensor_shape = (
			m,
			H // self.up_size[0],
			self.up_size[0],
			W // self.up_size[1],
			self.up_size[1],
			D // self.up_size[2],
			self.up_size[2],
			n_c_out
		)
		dx = np.reshape(da, tensor_shape).sum(axis=(2, 4, 6))

		return dx, np.array([[0]]), np.array([[0]])


# Other Extra Functionality


class Flatten(Layer):
	"""Flatten Layer.

	Flatten the output of the previous layer into a
	single feature vector.

	Equivalent to Reshape((-1,))
	"""

	__name__ = 'Flatten'

	def __init__(self, name=None):
		"""
		Args:
			name: name of  the layer.
		"""
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
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		return np.reshape(x, (-1,)+self.n_out)

	def diff(self, da):
		dx = np.reshape(da, (-1,)+self.n_in)
		return dx, np.array([[0]]), np.array([[0]])


class Activation(Layer):
	"""Activation Layer."""

	__name__ = 'Activation'

	def __init__(self, act, name=None):
		"""
		Args:
			act: Either an activation instance or string.
			name: name of  the layer.
		"""
		if isinstance(act, str):
			self.activation = activations.get(act)
		if not isinstance(act, activations.Activation()):
			raise WrongObjectError(act, activations.Activation())
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
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

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
		"""
		Args:
			name: name of  the layer.
		"""
		if not isinstance(n_out, tuple):
			raise WrongObjectError(n_out, tuple())

		num_of_unk_ch = 0
		self.unk_ch_id = None
		for i, ch in enumerate(n_out):
			if ch == -1 or ch is None:
				if num_of_unk_ch:
					raise InvalidShapeError(n_out)
				num_of_unk_ch += 1
				self.unk_ch_id = i
			else:
				if ch <= 0:
					raise InvalidShapeError(n_out)

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
		if x.shape[1:] != self.n_in:
			raise UnsupportedShapeError(x.shape, self.n_in)

		return np.reshape(x, (-1,)+self.n_out)

	def diff(self, da):
		dx = np.reshape(da, (-1,)+self.n_in)
		return dx, np.array([[0]]), np.array([[0]])
