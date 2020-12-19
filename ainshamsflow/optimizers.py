"""Optimizers Module.

In this Module, we include our optimization algorithms such as
Stocastic Gradient Descent (SGD).
"""

import numpy as np

from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError
from ainshamsflow.utils.history import History
#TODO: Add More Optimizers


def get(opt_name):
	"""Get any Optimizer in this Module by name."""

	opts = [SGD]
	for opt in opts:
		if opt.__name__.lower() == opt_name.lower():
			return opt()
	else:
		raise NameNotFoundError(opt_name, __name__)


class Optimizer:
	"""Optimiser Base Class.

	Used to generalize learning using Gradient Descent.

	Creating a new optimizer is as easy as creating a new class that
	inherits from this class.
	You then have to add any extra parameters in your constructor
	(while still calling this class' constructor) and redefine the
	_update() method.
	"""

	def __init__(self, lr=0.01):
		"""Initialize the Learning Rate."""
		self.lr = lr

	def __call__(self, x, y, epochs, batch_size, layers, loss, metrics, regularizer, training=True, verbose=True):
		"""Run optimization using Gradient Descent. Return History Object for training session."""

		m = x.shape[0]
		num_of_batches = int(m / batch_size)
		rem_batch_size = m - batch_size * num_of_batches
		history = History(loss, metrics)
		if verbose:
			print()
			print()
			if training:
				print('Training Model for {} epochs:'.format(epochs))
			else:
				print('Evaluating Model:')

		for epoch_num in range(epochs):
			loss_values = []
			metrics_values = []
			for i in range(num_of_batches):
				batch_x = x[i * batch_size: (i + 1) * batch_size]
				batch_y = y[i * batch_size: (i + 1) * batch_size]
				loss_value, metric_values = self._single_iteration(batch_x, batch_y, m, layers,
																   loss, metrics, regularizer, training)
				loss_values.append(loss_value)
				metrics_values.append(metric_values)

			if rem_batch_size:
				batch_x = x[-rem_batch_size:]
				batch_y = y[-rem_batch_size:]
				loss_value, metric_values = self._single_iteration(batch_x, batch_y, m, layers,
																   loss, metrics, regularizer, training)
				loss_values.append(loss_value)
				metrics_values.append(metric_values)
			loss_values = np.array(loss_values).sum()
			metrics_values = np.array(metrics_values).mean(axis=1)
			history.add(loss_values, metrics_values)
			if verbose:
				if training:
					print(
						'Finished epoch number {:4d}:'.format(epoch_num),
						'{}={:8.4f},'.format(loss.__name__, loss_values),
						*['{}={:8.4f},'.format(metric.__name__, metrics_values[j]) for j, metric in enumerate(metrics)]
					)
				else:
					print(
						'{}={:8.4f},'.format(loss.__name__, loss_values),
						*['{}={:8.4f},'.format(metric.__name__, metrics_values[j]) for j, metric in enumerate(metrics)]

					)

		return history

	def _single_iteration(self, batch_x, batch_y, m, layers, loss, metrics, regularizer, training):
		"""Run optimization for a single batch. Return loss value and metric values for iteration."""

		weights_list = [layer.get_weights()[0] for layer in layers]
		#Forward Pass
		batch_a = batch_x
		for j, layer in enumerate(layers):
			batch_a = layer(batch_a, training)
		regularization_term = 0 if regularizer is None else regularizer(weights_list, m)
		loss_value = loss(batch_a, batch_y) + regularization_term
		metric_values = []
		for metric in metrics:
			metric_values.append(metric(batch_a, batch_y))
		#Backward Pass
		regularization_diff = None if regularizer is None else regularizer.diff(weights_list, m)
		da = loss.diff(batch_a, batch_y)
		if training:
			for j in reversed(range(len(layers))):
				da, dw, db = layers[j].diff(da)
				if layers[j].trainable:
					if regularization_diff is not None:
						dw += regularization_diff[j]
					weights, biases = layers[j].get_weights()
					updated_weights = self._update(weights, dw)
					updated_biases = self._update(biases, db)
					layers[j].set_weights(updated_weights, updated_biases)
		return loss_value, metric_values

	def _update(self, weights, dw):
		raise BaseClassError


class SGD(Optimizer):
	"""Stochastic Gradient Descent Algorithm."""

	__name__ = 'SGD'

	def _update(self, weights, dw):
		"""Update step for SGD."""
		if isinstance(weights, list) and isinstance(dw, list):
			ans = []
			for weight, d in zip(weights, dw):
				ans.append(weight - self.lr * d)
			return ans
		else:
			assert np.shape(weights) == np.shape(dw)
			return weights - self.lr * dw
