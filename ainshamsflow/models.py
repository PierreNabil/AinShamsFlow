"""Models Module.

In this module we define the Sequential model type, which is the
main model type in asf.
We may implement more model types in the future.
"""

import shelve
from os.path import join as join_path
import numpy as np

import ainshamsflow.layers as _layers
import ainshamsflow.optimizers as optimizers
import ainshamsflow.losses as losses
import ainshamsflow.metrics as _metrics
import ainshamsflow.regularizers as regularizers
from ainshamsflow.data import Dataset
from ainshamsflow.utils.utils import get_dataset_from_xy
from ainshamsflow.utils.asf_errors import (UncompiledModelError, MultipleAccessError, BaseClassError,
										   LayerNotFoundError, UnsupportedShapeError, WrongObjectError,
										   InvalidShapeError)


__pdoc__ = dict()

__pdoc__['Model.evaluate'] = False
__pdoc__['Model.fit'] = False
__pdoc__['Model.get_layer'] = False
__pdoc__['Model.load_weights'] = False
__pdoc__['Model.pop_layer'] = False
__pdoc__['Model.add_layer'] = False
__pdoc__['Model.get_layer'] = False
__pdoc__['Model.load_weights'] = False
__pdoc__['Model.save_weights'] = False
__pdoc__['Model.print_summary'] = False

__pdoc__['Sequential.diff'] = False
__pdoc__['Sequential.add_input_shape_to_layers'] = False
__pdoc__['Sequential.set_weights'] = False
__pdoc__['Sequential.get_weights'] = False


def load_model(filename):
	"""Loads a model saved in 'filename'."""

	with shelve.open(filename) as db:
		model = db['self']
	return model


class Model(_layers.Layer):
	"""Models Base Class."""

	def __init__(self, input_shape, name, trainable=True):
		"""
		Args:
			input_shape: tuple showing the shape of the input to the model.
			name: Name of the model.
			trainable: Boolean to define whether this model is trainable or not.
		"""

		if not isinstance(input_shape, tuple):
			raise WrongObjectError(input_shape, tuple())
		if len(input_shape) <= 0:
			raise InvalidShapeError(input_shape)
		for ch in input_shape:
			if ch <= 0:
				raise InvalidShapeError(input_shape)

		super().__init__(name, trainable)
		self.n_in = input_shape
		self.n_out = None
		self.input_shape = '(None' + (',{:4}'*len(self.n_in)).format(*self.n_in) + ')'
		self.output_shape = None

		self.optimizer = None
		self.loss = None
		self.metrics = None
		self.regularizer = None

	def compile(self, optimizer, loss, metrics=None, regularizer=None):
		"""Define model optimizer, loss function , metrics and regularizer.

		All arguments can be either an instance of the required class,
		or a string for a class that exists in the framework.

		Args:
			optimizer: the optimizer used for training.
			loss: loss function used for training.
			metrics: a list of metrics (and/or losses) used to monitor training progress.
			regularizer: regularizer used during training.
		"""
		if isinstance(optimizer, str):
			optimizer = optimizers.get(optimizer)
		if not isinstance(optimizer, optimizers.Optimizer):
			raise WrongObjectError(optimizer, optimizers.Optimizer())

		if isinstance(loss, str):
			loss = losses.get(loss)
		if not isinstance(loss, losses.Loss):
			raise WrongObjectError(loss, losses.Loss())

		if metrics:
			if not isinstance(metrics, list):
				raise WrongObjectError(metrics, list())
			for i in range(len(metrics)):
				if isinstance(metrics[i], str):
					metrics[i] = _metrics.get(metrics[i])
				if not isinstance(metrics[i], _metrics.Metric):
					raise WrongObjectError(metrics[i], _metrics.Metric())
		else:
			metrics = []

		if regularizer is not None:
			if isinstance(regularizer, str):
				regularizer = regularizers.get(regularizer)
			if not isinstance(regularizer, regularizers.Regularizer):
				raise WrongObjectError(regularizer, regularizers.Regularizer())


		self.optimizer = optimizer
		self.loss = loss
		self.metrics = metrics
		self.regularizer = regularizer

	def evaluate(self, x, y, batch_size):
		raise BaseClassError

	def fit(self, x, y, epochs, batch_size):
		raise BaseClassError

	def predict(self, x):
		"""Predict new data.

		Args:
			x: input data to be passed through the model.
		"""

		if isinstance(x, Dataset):
			x = x.data

		return self.__call__(x)

	def add_layer(self, layer):
		raise BaseClassError

	def get_layer(self, layer_name, index):
		raise BaseClassError

	def pop_layer(self):
		raise BaseClassError

	def load_weights(self, filepath):
		raise BaseClassError

	def save_weights(self, filepath):
		raise BaseClassError

	def save_model(self, filename):
		"""Saves model by its name in the directory specified."""

		with shelve.open(filename) as db:
			db['self'] = self

	def print_summary(self):
		raise BaseClassError


class Sequential(Model):
	"""Sequential Model Class.
	Used to create Models where there is a strict, linear, layer-by-layer
	structure.
	"""

	__name__ = 'Seq. Model'

	def __init__(self, layers, input_shape, name=None):
		"""
		Args:
			layers: list of layers in order of data path.
				must be a list of layer objects.
			input_shape: tuple showing the shape of the input to the model.
			name: name of the model.
		"""

		if not isinstance(layers, list):
			raise WrongObjectError(layers, list())
		for layer in layers:
			if not isinstance(layer, _layers.Layer):
				raise WrongObjectError(layer, _layers.Layer())
		if not isinstance(input_shape, tuple):
			raise WrongObjectError(input_shape, tuple())

		super().__init__(input_shape, name)
		self.layers = layers

		for layer in self.layers:
			input_shape = layer.add_input_shape_to_layer(input_shape)

		self.n_out = self.layers[-1].n_out
		n_out = [str(ch) for ch in self.n_out]
		self.output_shape = '(None' + (',{:4}'*len(n_out)).format(*n_out) + ')'

	def fit(self, x, y=None, epochs=1, batch_size=None, verbose=True, live_plot=False, shuffle=True,
			valid_split=None, valid_data=None, valid_batch_size=None):
		"""Fit the model to the training data.

		Args:
			x: Either an Array of data features. must be of the same shape as the model's
				`input_shape`.
				__OR__ a dataset object that may include the data and labels for training.

			y: This is the array of data labels.
				Only used if `x` doesn't have the labels included in the dataset object.

			epochs: number of rounds gone through the training set during training.

			batch_size: size of the mini-batch used during training:
				`None` => batch training,
				`1` => online training,
				other int => mini-batch training.

			verbose: Whether to print training progress during training.

			live_plot: Whether to plot the training progress during training.

			shuffle: Whether to shuffle the training data before training.

			valid_split: how much to split the training set to get a new validation set.
				(0 < `valid_split` < 1)
				Valid only if `valid_data` is `None`

			valid_data: The validation set used in showing the training statistics.
				Either a tuple: (`x_valid`, `y_valid`) __OR__ a dataset object with data
				features and labels included.

			valid_batch_size: size of the mini-batch used during evaluation of the
				validation set:
				`None` => batch training,
				`1` => online training,
				other int => mini-batch training.

		Returns:
			history: history object used to plot the training statistics.
				you can plot the learning curves by using:

		```python
		>>> history = model.fit(...)
		>>> history.show()
		```
		"""

		if self.optimizer is None:
			raise UncompiledModelError

		ds_train = get_dataset_from_xy(x, y)

		if valid_data is not None:
			if isinstance(valid_data, tuple):
				ds_valid = get_dataset_from_xy(*valid_data)
			else:
				ds_valid = get_dataset_from_xy(valid_data, None)
		elif valid_split is not None:
			ds_train, ds_valid = ds_train.split(valid_split)
		else:
			ds_valid = None

		if shuffle:
			ds_train.shuffle()

		return self.optimizer(ds_train, ds_valid, epochs, batch_size, valid_batch_size, self.layers, self.loss,
							  self.metrics, self.regularizer, verbose=verbose, live_plot=live_plot, training=True)

	def evaluate(self, x, y=None, batch_size=None, verbose=True):
		"""Evaluate the model on validation data.

		Args:
			x: Either an Array of data features. must be of the same shape as the model's
				`input_shape`.
				__OR__ a dataset object that may include the data and labels for training.

			y: This is the array of data labels.
				Only used if `x` doesn't have the labels included in the dataset object.

			batch_size: size of the mini-batch used during evaluation:
				`None` => batch evaluation,
				`1` => online evaluation,
				other int => mini-batch evaluation.

			verbose: Whether to print the evaluation data or return it.

		Returns:
			None: if `verbose` = `True` __OR__
			loss_value: value of the loss function.
			metric_values: list of the metric values for all metrics.

		"""

		ds = get_dataset_from_xy(x, y)

		if self.optimizer is None:
			raise UncompiledModelError
		return self.optimizer(ds, None,  1, batch_size, None, self.layers, self.loss, self.metrics, self.regularizer,
								 verbose=verbose, live_plot=False, training=False)

	def add_layer(self, layer):
		"""Add a new layer to the network."""

		if not isinstance(layer, _layers.Layer):
			raise WrongObjectError(layer, _layers.Layer())

		if self.layers:
			n_out = self.layers[-1].n_out
		else:
			n_out = self.n_in
		self.layers.append(layer)
		layer.add_input_shape_to_layer(n_out)

	def get_layer(self, *, index=None, layer_name=None):
		"""Get a specific layer from the model.

		You can either access the layer using it's name __xor__ it's index.
		"""

		if (index is None) ^ (layer_name is None):
			if not isinstance(index, int):
				raise WrongObjectError(index, 1)
			if not isinstance(layer_name, str):
				raise WrongObjectError(layer_name, '')

			if index is None:
				for layer in self.layers:
					if layer.name == layer_name:
						return layer
				raise LayerNotFoundError('name', layer_name)
			elif index < len(self.layers):
				return self.layers[index]
			else:
				raise LayerNotFoundError('id', index)
		else:
			raise MultipleAccessError

	def pop_layer(self):
		"""Pop the last layer from the model."""

		if self.layers:
			self.layers.pop()

	def load_weights(self, filepath):
		"""Load weights from a directory."""

		with shelve.open(join_path(filepath, self.name+'_weights')) as db:
			for i in range(len(self.layers)):
				layer_name = 'layer{:03d}'.format(i)
				self.layers[i] = db[layer_name]

	def save_weights(self, filepath):
		"""Save weights to a directory."""

		with shelve.open(join_path(filepath, self.name+'_weights')) as db:
			for i, layer in enumerate(self.layers):
				layer_name = 'layer{:03d}'.format(i)
				db[layer_name] = layer.get_weights()

	def print_summary(self):
		"""Print a summary of the sequential model in a table."""

		spacer = '+' + '-' * 22 + '+' + '-' * 15 + '+' + '-' * 23 + '+' + '-' * 23 + '+' + '-' * 32 + '+'
		total_params = self.count_params()
		trainable_params = self.count_params(trainable_only=True)

		print()
		print('Sequential Model: {}'.format(self.name))
		print(spacer)
		print('| {:20s} | {:13s} | {:>21s} | {:>21s} | {:30s} |'.format('Layer Type:', 'num_of_params',
														'input_shape', 'output_shape', 'layer_name'))
		print(spacer)
		for layer in self.layers:
			print('| ', layer.summary(), ' |', sep='')
		print(spacer)
		print('| {:117s} |'.format('Total Params: {}'.format(total_params)))
		print('| {:117s} |'.format('Trainable Params: {}'.format(trainable_params)))
		print('| {:117s} |'.format('Non-Trainable Params: {}'.format(total_params - trainable_params)))
		print(spacer)
		print()

	# Model as a Layer Functionality:

	def __call__(self, x, training=False):
		a = x
		for layer in self.layers:
			a = layer(a, training)
		return a

	def diff(self, da):
		Dw = []
		Db = []

		for layer in reversed(self.layers):
			da, dw, db = layer.diff(da)
			Dw.insert(0, dw)
			Db.insert(0, db)

		return da, Dw, Db

	def add_input_shape_to_layers(self, n_in):
		if n_in != self.n_in:
			raise UnsupportedShapeError(n_in, self.n_in)
		if self.layers:
			return self.layers[-1].n_out

	def count_params(self, trainable_only=False):
		if trainable_only:
			return np.sum([layer.count_params() for layer in self.layers if layer.trainable])
		else:
			return np.sum([layer.count_params() for layer in self.layers])

	def get_weights(self):
		weights = []
		biases  = []
		for layer in self.layers:
			w, b = layer.get_weights()
			weights.append(w)
			biases.append(b)
		return weights, biases

	def set_weights(self, weights, biases):
		for i, layer in enumerate(self.layers):
			if layer.trainable:
				layer.set_weights(weights[i], biases[i])
