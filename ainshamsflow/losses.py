"""Losses Module.

In this Module, we include our loss functions such as the
Mean Square Error (MSE) or Binary Cross Entropy.
"""

import numpy as np

from ainshamsflow.metrics import Metric
from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError
#TODO: Add More Losses


def get(loss_name):
	"""Get any Loss in this Module by name."""

	losses = [MSE, MAE, MAPE]
	for loss in losses:
		if loss.__name__.lower() == loss_name.lower() :
			return loss()
	raise NameNotFoundError(loss_name, __name__)


class Loss(Metric):
	"""Loss Base Class.

	To create a new Loss Function, create a class that inherits
	from this class.
	You then have to add any parameters in your constructor
	and redefine the __call__() and diff() methods.

	Note: all loss functions can be used as metrics.
	"""

	def diff(self, y_pred, y_true):
		raise BaseClassError


class MSE(Loss):
	"""Mean Squared Error Loss Function."""

	__name__ = 'MSE'

	def __call__(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = np.sum(y_true.shape[0])
		return np.sum(np.square(y_pred - y_true)) / (2*m)

	def diff(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = np.sum(y_true.shape[0])
		return (y_pred - y_true) / m


class MAE(Loss):
	"""Mean Absolute Error Loss Function."""
	__name__ = 'MAE'

	def __call__(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = np.sum(y_true.shape[0])
		return np.sum(np.abs(y_pred - y_true)) / m

	def diff(self, y_pred, y_true):
		assert y_true.shape == y_pred.shape
		m = np.sum(y_true.shape[0])
		return np.where(y_pred > y_true, 1, -1) / m
