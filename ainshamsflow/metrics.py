import numpy as np

from ainshamsflow.utils.utils import pred_one_hot, true_one_hot
from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError, UnsupportedShapeError


def get(metric_name):
	metrics = [Accuracy, Precision, Recall, F1Score,
			   TruePositive, TrueNegative, FalsePositive, FalseNegative]
	for metric in metrics:
		if metric.__name__.lower() == metric_name.lower():
			return metric()
	raise NameNotFoundError(metric_name, __name__)


def _check_dims(y_pred, y_true):
	if y_true.shape[-1] == 1 and y_pred.shape[-1] != 1:
		n_c = y_pred.shape[-1]
		y_true = true_one_hot(y_true, n_c)
	if y_true.shape != y_pred.shape:
		raise UnsupportedShapeError(y_true.shape, y_pred.shape)
	return y_true


class Metric:
	def __call__(self, y_pred, y_true):
		raise BaseClassError


class FalseNegative(Metric):
	__name__ = 'FN'

	def __call__(self, y_pred, y_true):
		y_true = _check_dims(y_pred, y_true)
		y_pred = pred_one_hot(y_pred)
		return np.sum(np.logical_and(y_pred == 0, y_true == 1))


class FalsePositive(Metric):
	__name__ = 'FP'

	def __call__(self, y_pred, y_true):
		y_true = _check_dims(y_pred, y_true)
		y_pred = pred_one_hot(y_pred)
		return np.sum(np.logical_and(y_pred == 1, y_true == 0))


class TrueNegative(Metric):
	__name__ = 'TN'

	def __call__(self, y_pred, y_true):
		y_true = _check_dims(y_pred, y_true)
		y_pred = pred_one_hot(y_pred)
		return np.sum(np.logical_and(y_pred == 0, y_true == 0))


class TruePositive(Metric):
	__name__ = 'TP'

	def __call__(self, y_pred, y_true):
		y_true = _check_dims(y_pred, y_true)
		y_pred = pred_one_hot(y_pred)
		return np.sum(np.logical_and(y_pred == 1, y_true == 1))


class Accuracy(Metric):
	__name__ = 'Accuracy'

	def __call__(self, y_pred, y_true):
		TP = TruePositive()(y_pred, y_true)
		TN = TrueNegative()(y_pred, y_true)
		FP = FalsePositive()(y_pred, y_true)
		FN = FalseNegative()(y_pred, y_true)
		return (TP + TN) / (TP + TN + FP + FN)


class Precision(Metric):
	__name__ = 'Precision'

	def __call__(self, y_pred, y_true):
		TP = TruePositive()(y_pred, y_true)
		FP = FalsePositive()(y_pred, y_true)
		return TP/(TP+FP)


class Recall(Metric):
	__name__ = 'Recall'

	def __call__(self, y_pred, y_true):
		TP = TruePositive()(y_pred, y_true)
		FN = FalseNegative()(y_pred, y_true)
		return TP/(TP+FN)


class F1Score(Metric):
	__name__ = 'F1Score'

	def __call__(self, y_pred, y_true):
		TP = TruePositive()(y_pred, y_true)
		FN = FalseNegative()(y_pred, y_true)
		FP = FalsePositive()(y_pred, y_true)
		return 2*TP/(2*TP+FP+FN)
