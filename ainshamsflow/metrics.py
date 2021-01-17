import numpy as np

from ainshamsflow.utils.utils import pred_one_hot, true_one_hot, confution_matrix
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
	def __call__(self):
		raise BaseClassError

	@classmethod
	def calc_confusion_matrix(cls, y_pred, y_true):
		y_true = _check_dims(y_pred, y_true)
		y_pred = pred_one_hot(y_pred)
		cls.conf_mtx = confution_matrix(y_pred, y_true)
		cls.m = y_pred.shape[0]


class FalseNegative(Metric):
	__name__ = 'FN'

	def class_wise(self):
		return np.sum(self.conf_mtx, axis=1) - TruePositive().class_wise()

	def __call__(self):
		return np.sum(self.class_wise())


class FalsePositive(Metric):
	__name__ = 'FP'

	def class_wise(self):
		return np.sum(self.conf_mtx, axis=0) - TruePositive().class_wise()

	def __call__(self):
		return np.sum(self.class_wise())


class TrueNegative(Metric):
	__name__ = 'TN'

	def class_wise(self):
		n = self.conf_mtx.shape[0]
		return np.array([np.sum(np.delete(np.delete(self.conf_mtx, i, 0), i, 1)) for i in range(n)])

	def __call__(self):
		return np.sum(self.class_wise())


class TruePositive(Metric):
	__name__ = 'TP'

	def class_wise(self):
		n = self.conf_mtx.shape[0]
		return np.array([self.conf_mtx[i][i] for i in range(n)])

	def __call__(self):
		return np.sum(self.class_wise())


class Accuracy(Metric):
	__name__ = 'Accuracy'

	def __call__(self):
		return TruePositive()() / self.m


class Precision(Metric):
	__name__ = 'Precision'

	def __call__(self):
		TP = TruePositive().class_wise()
		FP = FalsePositive().class_wise()
		return np.mean(np.where(np.logical_and(TP == 0, FP == 0), 0, TP/(TP+FP)))


class Recall(Metric):
	__name__ = 'Recall'

	def __call__(self):
		TP = TruePositive().class_wise()
		FN = FalseNegative().class_wise()
		return np.mean(np.where(np.logical_and(TP == 0, FN == 0), 0, TP/(TP+FN)))


class F1Score(Metric):
	__name__ = 'F1Score'

	def __call__(self):
		p = Precision()()
		r = Recall()()
		return 2 / (1/p + 1/r)
