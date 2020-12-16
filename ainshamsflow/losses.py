import numpy as np

from ainshamsflow.utils.asf_errors import BaseClassError
#TODO: Add More Losses


class Loss:
	def __call__(self, y_pred, y_true):
		raise BaseClassError

	def diff(self, y_pred, y_true):
		raise BaseClassError


class MSE(Loss):
	name = 'MSE'

	def __call__(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = np.sum(y_true.shape[1])
		return np.sum(np.square(y_pred - y_true), axis=1, keepdims=True) / (2*m)

	def diff(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = np.sum(y_true.shape[1])
		return (y_pred - y_true) / m


class MAE(Loss):
	name = 'MAE'

	def __call__(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = np.sum(y_true.shape[1])
		return np.sum(np.abs(y_pred - y_true), axis=1, keepdims=True) / m

	def diff(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = np.sum(y_true.shape[1])
		return np.where(y_pred > y_true, 1, -1) / m
