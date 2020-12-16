import numpy as np

from ainshamsflow.utils.peras_errors import BaseClassError
#TODO: Add More Metrics


class Metric:
	def __call__(self, y_pred, y_true):
		raise BaseClassError


class HardAccuracy(Metric):
	name = 'HardAccuracy'

	def __call__(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		return np.count(y_pred == y_true)


class SoftAccuracy(Metric):
	name = 'SoftAccuracy'

	def __call__(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		return 1 - np.sum(np.abs(y_pred - y_true)/y_true) / np.count(y_pred)
