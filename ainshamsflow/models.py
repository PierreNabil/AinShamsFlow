"""Models Module.

In this module we define the Sequential model type, which is the
main model type in asf.
We may implement more model types in the future.
"""


import shelve
from os.path import join as join_path
import numpy as np

from ainshamsflow.layers import Layer
from ainshamsflow.optimizers import Optimizer
from ainshamsflow.losses import Loss
from ainshamsflow.metrics import Metric
from ainshamsflow.data import Dataset
from ainshamsflow.utils.utils import get_dataset_from_xy
from ainshamsflow.utils.asf_errors import (UncompiledModelError, MultipleAccessError, BaseClassError,
										   LayerNotFoundError, UnsupportedShapeError, RunningWithoutDataError)


def load_model(filename):
	"""Loads a model saved in 'filename'."""

	with shelve.open(filename) as db:
		model = db['self']
	return model


class Model(Layer):
	"""Models Base Class."""

	def __init__(self, input_shape, name, trainable=True):
		"""Initialize model name."""

		assert isinstance(input_shape, tuple)
		assert len(input_shape) > 0
		for ch in input_shape:
			assert ch > 0

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
		"""Define model optimizer, loss function , metrics and regularizer."""

		if metrics is None:
			metrics = []
		assert isinstance(optimizer, Optimizer)
		assert isinstance(loss, Loss)
		if metrics:
			assert isinstance(metrics, list)
			for metric in metrics:
				assert isinstance(metric, Metric)

		self.optimizer = optimizer
		self.loss = loss
		self.metrics = metrics
		self.regularizer = regularizer

	def evaluate(self, x, y, batch_size):
		raise BaseClassError

	def fit(self, x, y, epochs, batch_size):
		raise BaseClassError

	def predict(self, x):
		"""Predict new data."""

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

	def save_model(self, filepath=''):
		"""Saves model by its name in the directory specified."""

		with shelve.open(join_path(filepath, self.name)) as db:
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
		"""Define the model layers, input shape and name."""

		assert isinstance(layers, list)
		for layer in layers:
			assert isinstance(layer, Layer)
		assert isinstance(input_shape, tuple)

		super().__init__(input_shape, name)
		self.layers = layers

		for layer in self.layers:
			input_shape = layer.add_input_shape_to_layers(input_shape)

		self.n_out = self.layers[-1].n_out
		self.output_shape = '(None' + (',{:4}'*len(self.n_out)).format(*self.n_out) + ')'

	def fit(self, x, y=None, epochs=1, batch_size=None, verbose=True):
		"""Fit the model to the training data."""

		ds = get_dataset_from_xy(x, y)

		if self.optimizer is None:
			raise UncompiledModelError
		return self.optimizer(ds, epochs, batch_size, self.layers, self.loss, self.metrics, self.regularizer,
							  verbose=verbose, training=True)

	def evaluate(self, x, y=None, batch_size=None, verbose=True):
		"""Evaluate the model on validation data."""

		ds = get_dataset_from_xy(x, y)

		if self.optimizer is None:
			raise UncompiledModelError
		history = self.optimizer(ds, 1, batch_size, self.layers, self.loss, self.metrics, self.regularizer,
								 verbose=verbose, training=False)
		loss_value = np.mean(history.loss_values)
		metric_values = history.flipped_metrics()
		return loss_value, metric_values

	def add_layer(self, layer):
		"""Add a new layer to the network."""

		assert isinstance(layer, Layer)
		if self.layers:
			n_out = self.layers[-1].n_out
		else:
			n_out = self.n_in
		self.layers.append(layer)
		layer.add_input_shape_to_layer(n_out)

	def get_layer(self, *, index=None, layer_name=None):
		"""Get a specific layer from the model."""

		if (index is None) ^ (layer_name is None):
			assert isinstance(index, int) or isinstance(layer_name, str)
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
			a = layer(a)
		return a

	def diff(self, da):
		Dw = []
		Db = []

		for layer in reversed(self.layers):
			da, dw, db = layer.diff(da)
			if layer.trainable:
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
		w_and_b = np.array([layer.get_weights() for layer in self.layers if layer.trainable])
		weights = list(w_and_b[:, 0])
		biases  = list(w_and_b[:, 1])
		return weights, biases

	def set_weights(self, weights, biases):
		for i, layer in enumerate(self.layers):
			if layer.trainable:
				layer.set_weights(weights[i], biases[i])
