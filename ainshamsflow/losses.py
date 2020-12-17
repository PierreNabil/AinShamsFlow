import numpy as np

from ainshamsflow.metrics import Metric
from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError
#TODO: Add More Losses


def get(loss_name):
	losses = [MSE, MAE, MAPE]
	for loss in losses:
		if loss.__name__.lower() == loss_name.lower() :
			return loss()
	raise NameNotFoundError(loss_name, __name__)


class Loss(Metric):
	def diff(self, y_pred, y_true):
		raise BaseClassError


class MSE(Loss):
	__name__ = 'MSE'

	def __call__(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = np.sum(y_true.shape[1])
		return np.sum(np.square(y_pred - y_true), axis=1, keepdims=True) / (2*m)

	def diff(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = np.sum(y_true.shape[1])
		return (y_pred - y_true) / m


class MAE(Loss):
	__name__ = 'MAE'

	def __call__(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = np.sum(y_true.shape[1])
		return np.sum(np.abs(y_pred - y_true), axis=1, keepdims=True) / m

	def diff(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = np.sum(y_true.shape[1])
		return np.where(y_pred > y_true, 1, -1) / m


class MAPE(Loss):
	__name__ = 'MAPE'

	def __call__(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = y_true.shape[1]
		return 1 - np.sum(np.abs(y_pred - y_true)/y_true) / m

	def diff(self, y_pred, y_true):
		#Todo:
		pass