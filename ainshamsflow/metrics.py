import numpy as np

from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError, UnsupportedShapeError


def get(metric_name):
	metrics = [Accuracy, Precision, Recall, F1score,
			   TruePositives, TrueNegatives, FalsePositives, FalseNegatives]
	for metric in metrics:
		if metric.__name__.lower() == metric_name.lower():
			return metric
	raise NameNotFoundError(metric_name, __name__)


def _one_hot(y_pred):
	n_c = y_pred.shape[-1]
	return np.squeeze(np.eye(n_c)[np.argmax(y_pred, axis=-1)])


class Metric:
	def __call__(self, y_pred, y_true):
		raise BaseClassError


class Accuracy(Metric):
	__name__ = 'Accuracy'

	def __call__(self, y_pred, y_true):
		if y_true.shape != y_pred.shape:
			raise UnsupportedShapeError(y_pred, y_true)
		m = y_true.shape[0]
		y_pred = _one_hot(y_pred)
		return np.sum(y_pred == y_true) / m


class FalseNegatives(Metric):
	def __call__(self, y_true, y_pred):
		if y_true.shape != y_pred.shape :
			raise UnsupportedShapeError(y_pred, y_true)
		y_pred = _one_hot(y_pred)
		return np.sum(np.logical_and(y_pred == 0, y_true == 1))


class FalsePositives(Metric):
	def __call__(self, y_true, y_pred):
		if y_true.shape != y_pred.shape :
			raise UnsupportedShapeError(y_pred, y_true)
		y_pred = _one_hot(y_pred)
		return np.sum(np.logical_and(y_pred == 1, y_true == 0))


class TrueNegatives(Metric):
	def __call__(self, y_true, y_pred):
		if y_true.shape != y_pred.shape :
			raise UnsupportedShapeError(y_pred, y_true)
		y_pred = _one_hot(y_pred)
		return np.sum(np.logical_and(y_pred == 0, y_true == 0))


class TruePositives(Metric):
	def __call__(self, y_true, y_pred):
		if y_true.shape != y_pred.shape :
			raise UnsupportedShapeError(y_pred, y_true)
		y_pred = _one_hot(y_pred)
		return np.sum(np.logical_and(y_pred == 1, y_true == 1))


class Precision(Metric):
	def __call__(self, y_true, y_pred):
		if y_true.shape != y_pred.shape :
			raise UnsupportedShapeError(y_pred, y_true)
		TP = TruePositives()(y_true, y_pred)
		FP = FalsePositives()(y_true, y_pred)
		return TP/(TP+FP)


class Recall(Metric):
	def __call__(self, y_true, y_pred):
		if y_true.shape != y_pred.shape :
			raise UnsupportedShapeError(y_pred, y_true)
		TP = TruePositives()(y_true, y_pred)
		FN = FalseNegatives()(y_true, y_pred)
		return TP/(TP+FN)


class F1score(Metric):
	def __call__(self, y_true, y_pred):
		if y_true.shape != y_pred.shape :
			raise UnsupportedShapeError(y_pred, y_true)
		TP = TruePositives()(y_true, y_pred)
		FN = FalseNegatives()(y_true, y_pred)
		FP = FalsePositives()(y_true)
		return 2*TP/(2*TP+FP+FN)
