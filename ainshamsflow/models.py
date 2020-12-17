import shelve
import os
import numpy as np

from ainshamsflow.layers import Layer
from ainshamsflow.optimizers import Optimizer
from ainshamsflow.losses import Loss
from ainshamsflow.metrics import Metric
from ainshamsflow.utils.asf_errors import UncompiledModelError, MultipleAccessError, BaseClassError


def load_model(filename):
	with shelve.open(filename) as db:
		model = db['self']
	return model


class Model:
	def __init__(self, input_shape, name):
		assert isinstance(input_shape[0], int)
		assert isinstance(name, str)

		self.input_shape = (input_shape[0], None)
		self.name = name

		self.optimizer = None
		self.loss = None
		self.metrics = None
		self.regularizer = None

	def compile(self, optimizer, loss, metrics=[], regularizer=None):
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

	def save_model(self, filepath):
		with shelve.open(os.path.join(filepath, self.name + '.h5')) as db:
			db['self'] = self

	def print_summary(self):
		raise BaseClassError


class Sequential(Model):
	def __init__(self, layers, input_size, name):
		assert isinstance(layers, list)
		for layer in layers:
			assert isinstance(layer, Layer)
		assert isinstance(input_size, int)

		input_shape = (input_size, None)
		for layer in layers:
			input_size = layer.add_input_shape_to_layers(input_size)

		super().__init__(input_shape, name)
		self.layers = layers

	def evaluate(self, x, y, batch_size=None, verbose=True):
		if self.optimizer is None:
			raise UncompiledModelError
		if batch_size is None:
			m = x.shape[1]
			batch_size = m
		history = self.optimizer(x, y, 1, batch_size, self.layers, self.loss, self.metrics, self.regularizer,
								 verbose=verbose, training=False)
		loss_value = np.mean(history.loss_values)
		metric_values = history.fliped_metrics()
		return loss_value, metric_values

	def fit(self, x, y, epochs, batch_size=None, verbose=True):
		if self.optimizer is None:
			raise UncompiledModelError
		if batch_size is None:
			m = x.shape[1]
			batch_size = m
		return self.optimizer(x, y, epochs, batch_size, self.layers, self.loss, self.metrics, self.regularizer,
							  verbose=verbose, training=True)

	def predict(self, x):
		a = x
		for layer in self.layers:
			a = layer(a)
		return a

	def add_layer(self, layer):
		assert isinstance(layer, Layer)
		self.layers.append(layer)

	def get_layer(self, index=None, layer_name=None):
		if (index is None) ^ (layer_name is None):
			if index is None:
				for layer in self.layers:
					if layer.name == layer_name:
						return layer
			else:
				return self.layers[index]
		else:
			raise MultipleAccessError

	def pop_layer(self):
		self.layers.pop()

	def load_weights(self, filepath):
		with shelve.open(os.path.join(filepath, self.name+'_weights')) as db:
			for i in range(len(self.layers)):
				layer_name = 'layer{:03d}'.format(i)
				self.layers[i] = db[layer_name]

	def save_weights(self, filepath):
		with shelve.open(os.path.join(filepath, self.name+'_weights')) as db:
			for i, layer in enumerate(self.layers):
				layer_name = 'layer{:03d}'.format(i)
				db[layer_name] = layer.get_weights()

	def print_summary(self):
		print()
		print('Sequential Model: {}'.format(self.name))
		print('_' * (20 + 13 + 11 + 11 + 20 + 4))
		print('{:20s} | {:13s} | {:11s} | {:11s} | {:s}'.format('Layer Type:', 'num_of_params',
														'input_shape', 'output_shape', 'layer_name'))
		print('_' * (20 + 13 + 11 + 11 + 20 + 4))
		# '{:20s} {:13d} {:11} {:11} {}'.format(layer_name, self.count_params(), input_shape, output_shape, self.name)
		for layer in self.layers:
			print(layer.summary())
		print('_' * (20 + 13 + 11 + 11 + 20 + 4))
