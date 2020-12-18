"""Metrics Module.

In this Module, we include our loss functions such as Accuracy.
"""

import numpy as np

from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError
#TODO: Add More Metrics


def get(metric_name):
	"""Get any Metric in this Module by name."""

	metrics = [Accuracy]
	for metric in metrics:
		if metric.__name__.lower() == metric_name.lower():
			return metric
	raise NameNotFoundError(metric_name, __name__)


class Metric:
	"""Metrics Base Class.

	To create a new Metric, create a class that inherits from
	this class.
	You then have to add any parameters in your constructor
	and redefine the __call__() method.
	"""

	def __call__(self, y_pred, y_true):
		raise BaseClassError


class Accuracy(Metric):
	"""Accuracy Metric."""

	__name__ = 'Accuracy'

	def __call__(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = y_true.shape[0]
		return np.sum(y_pred == y_true) / m
