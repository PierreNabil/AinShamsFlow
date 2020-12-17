import numpy as np

from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError
#TODO: Add More Metrics


def get(metric_name):
	metrics = [Accuracy]
	for metric in metrics:
		if metric.__name__.lower() == metric_name.lower():
			return metric
	raise NameNotFoundError(metric_name, __name__)


class Metric:
	def __call__(self, y_pred, y_true):
		raise BaseClassError


class Accuracy(Metric):
	__name__ = 'Accuracy'

	def __call__(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = y_true.shape[1]
		return np.sum(y_pred == y_true) / m
