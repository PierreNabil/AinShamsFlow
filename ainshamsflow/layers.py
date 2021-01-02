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
			  # Conv1D, Pool1D, GlobalPool1D, Upsample1D,
			  Conv2D, Pool2D, GlobalPool2D,  Upsample2D,
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
		assert isinstance(weights_init, initializers.Initializer)
		assert isinstance(biases_init, initializers.Initializer)

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


class Upsample1D(Layer):
	"""1-Dimensional Up Sampling Layer."""

	__name__ = 'Upsample1D'
	pass


class Conv2D(Layer):
	"""2-Dimensional Convolution Layer."""

	__name__ = 'Conv2D'

	def __init__(self, filters, kernel_size, strides=1, padding='valid',
				 kernel_init=initializers.Constant(1), biases_init=initializers.Constant(0),
				 activation=activations.Linear(), name=None, trainable=True):
		assert isinstance(filters, int)
		if isinstance(kernel_size, int):
			kernel_size = (kernel_size, kernel_size)
		assert isinstance(kernel_size, tuple)
		assert len(kernel_size) == 2
		for ch in kernel_size:
			assert ch > 0
			assert ch % 2 == 1
		if isinstance(strides, int):
			strides = (strides, strides)
		assert isinstance(strides, tuple)
		assert len(strides) == 2
		for ch in strides:
			assert ch > 0
		padding = padding.lower()
		assert padding == 'valid' or padding == 'same'
		assert isinstance(kernel_init, initializers.Initializer)
		assert isinstance(biases_init, initializers.Initializer)
		if isinstance(activation, str):
			activation = activations.get(activation)
		assert isinstance(activation, activations.Activation)

		super().__init__(name, trainable)
		self.n_in = (None, None, 1)
		self.n_out = (None, None, filters)
		self.kernel_init = kernel_init
		self.biases_init = biases_init
		self.activation = activation
		self.strides = strides
		self.same_padding = padding == 'same'

		self.input_shape = None
		self.output_shape = None

		self.kernel = kernel_size
		self.biases = None

	def add_input_shape_to_layers(self, n_in):
		assert len(n_in) == 3
		assert n_in[-1] > 0

		self.n_in = n_in
		self.kernel = self.kernel_init((self.n_out[-1], *self.kernel, self.n_in[-1]))
		self.biases = self.biases_init((self.n_out[-1], 1, 1, 1))

		self.input_shape = '(None,None,None,{:4d})'.format(self.n_in[-1])
		self.output_shape = '(None,None,None,{:4d})'.format(self.n_out[-1])

		return self.n_out

	def __call__(self, x, training=False):
		n_c_out, f_h, f_w, n_c_in = self.kernel.shape
		m, H_prev, W_prev, n_c_in = x.shape
		assert n_c_in == self.n_in[-1]
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
		assert n_c_out == self.n_out[-1]

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
		assert weights.shape == self.kernel.shape
		assert biases.shape == self.biases.shape

		self.kernel = np.array(weights)
		self.biases = np.array(biases)


class Pool2D(Layer):
	"""2-Dimensional Pooling Layer."""

	__name__ = 'Pool2D'

	def __init__(self, pool_size=2, strides=None, padding='valid', mode='max', name=None, trainable=True):
		if isinstance(pool_size, int):
			pool_size = (pool_size, pool_size)
		assert isinstance(pool_size, tuple)
		for ch in pool_size:
			assert ch > 0
		if strides is None:
			strides = pool_size
		elif isinstance(strides, int):
			strides = (strides, strides)
		assert isinstance(strides, tuple)
		for ch in strides:
			assert ch > 0
		padding = padding.lower()
		assert padding == 'valid' or padding == 'same'
		assert mode == 'max' or mode == 'avg' or mode == 'average' or mode == 'mean'
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
		assert len(n_in) == 3
		assert n_in[-1] > 0

		self.n_in = self.n_out = (None, None, n_in[-1])
		self.input_shape = self.output_shape = '(None,None,None,{:4d})'.format(n_in[-1])

		return n_in

	def __call__(self, x, training=False):
		f_h, f_w = self.pool_size
		m, H_prev, W_prev, n_c = x.shape
		assert n_c == self.n_in[-1]
		self.x = x

		p_h, p_w = 0, 0
		s_h, s_w = self.strides
		if self.same_padding :
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

		self.z = z
		return z

	def diff(self, da):
		f_h, f_w = self.pool_size
		m, H, W, n_c_out = da.shape
		assert n_c_out == self.n_out[-1]

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
								mask * da[:, vert_start:vert_end, horiz_start:horiz_end, c]
						)

					else:
						da_mean = np.mean(da[:, vert_start:vert_end, horiz_start:horiz_end, c], axis=(1,2))
						dx[:, vert_start :vert_end, horiz_start :horiz_end, c] += (
							da_mean[:, None, None] / np.prod(self.pool_size) * np.ones(self.pool_size)[None, ...]
						)

		return dx, np.array([[0]]), np.array([[0]])


class GlobalPool2D(Layer):
	"""Global Average Pooling Layer."""

	__name__ = 'GlobalAvgPool2D'
	pass


class Upsample2D(Layer):
	"""2-Dimensional Up Sampling Layer."""

	__name__ = 'Upsample2D'
	pass


# Other Extra Functionality


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
