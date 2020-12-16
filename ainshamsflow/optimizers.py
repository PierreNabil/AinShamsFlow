import numpy as np

from ainshamsflow.utils.peras_errors import BaseClassError
from ainshamsflow.utils.history import History
#TODO: Add More Optimizers


class Optimizer:
	def __init__(self, learning_rate):
		self.lr = learning_rate

	def __call__(self, x, y, epochs, batch_size, layers, loss, metrics, regularizer, training=True):
		m = x.shape[1]
		num_of_batches = np.floor(m / batch_size)
		rem_batch_size = m - batch_size * num_of_batches
		history = History(loss, metrics)

		for _ in range(epochs):
			for i in range(num_of_batches):
				batch_x = x[i * batch_size: (i + 1) * batch_size]
				batch_y = y[i * batch_size: (i + 1) * batch_size]
				loss_value, metric_values = self._single_iteration(batch_x, batch_y, m, layers,
																   loss, metrics, regularizer, training)
				history.add(loss_value, metric_values)

			if rem_batch_size:
				batch_x = x[-rem_batch_size:]
				batch_y = y[-rem_batch_size:]
				loss_value, metric_values = self._single_iteration(batch_x, batch_y, m, layers,
																   loss, metrics, regularizer, training)
				history.add(loss_value, metric_values)
		return history

	def _single_iteration(self, batch_x, batch_y, m, layers, loss, metrics, regularizer, training):
		weights_list = [layer.get_weights() for layer in layers]
		#Forward Pass
		batch_a = [batch_x]
		for j, layer in enumerate(layers):
			batch_a.append(layer(batch_a[j]))
		batch_a.pop(0)
		loss_value = loss(batch_a[-1], batch_y) + regularizer(weights_list, m)
		metric_values = []
		for metric in metrics:
			metric_values.append(metric(batch_a[-1], batch_y))
		#Backward Pass
		da = loss_value
		if training:
			for j in reversed(range(len(layers))):
				if layers[j].trainable:
					da, dw, db = layers[j].diff(da)
					weights, biases = weights_list[j]
					updated_weights = self._update(weights, dw)
					updated_biases = self._update(biases, db)
					layers[j].set_weights(updated_weights, updated_biases)
		return loss_value, metric_values

	def _update(self, weights, dw):
		raise BaseClassError


class SGD(Optimizer):
	def _update(self, weights, dw):
		assert weights.shape == dw.shape
		return weights - self.lr * dw
