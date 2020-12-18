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
from ainshamsflow.utils.asf_errors import UncompiledModelError, MultipleAccessError, BaseClassError, LayerNotFoundError


def load_model(filename):
	"""Loads a model saved in 'filename'."""

	with shelve.open(filename) as db:
		model = db['self']
	return model


class Model:
	"""Models Base Class."""

	def __init__(self, input_shape, name):
		"""Initialize model name."""

		assert isinstance(input_shape, tuple)
		assert len(input_shape) > 0
		for ch in input_shape:
			assert ch > 0

		self.name = str(name)
		self.input_shape = input_shape

		self.optimizer = None
		self.loss = None
		self.metrics = None
		self.regularizer = None

	def compile(self, optimizer, loss, metrics=[], regularizer=None):
		"""Define model optimizer, loss function , metrics and regularizer."""

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
		raise BaseClassError

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

	def fit(self, x, y, epochs, batch_size=None, verbose=True):
		"""Fit the model to the training data."""

		if self.optimizer is None:
			raise UncompiledModelError
		if batch_size is None:
			m = x.shape[0]
			batch_size = m
		return self.optimizer(x, y, epochs, batch_size, self.layers, self.loss, self.metrics, self.regularizer,
							  verbose=verbose, training=True)

	def evaluate(self, x, y, batch_size=None, verbose=True):
		"""Evaluate the model on validation data."""

		if self.optimizer is None:
			raise UncompiledModelError
		if batch_size is None:
			m = x.shape[0]
			batch_size = m
		history = self.optimizer(x, y, 1, batch_size, self.layers, self.loss, self.metrics, self.regularizer,
								 verbose=verbose, training=False)
		loss_value = np.mean(history.loss_values)
		metric_values = history.flipped_metrics()
		return loss_value, metric_values

	def predict(self, x):
		"""Predict new data."""

		a = x
		for layer in self.layers:
			a = layer(a)
		return a

	def add_layer(self, layer):
		"""Add a new layer to the network."""

		assert isinstance(layer, Layer)
		if self.layers:
			n_out = self.layers[-1].n_out
		else:
			n_out = self.input_shape
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
		total_params = np.sum([layer.count_params() for layer in self.layers])
		trainable_params = np.sum([layer.count_params() for layer in self.layers if layer.trainable])
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
