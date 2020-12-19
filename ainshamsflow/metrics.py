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


class FalseNegatives(Metric):
	def __call__(self, y_true,y_pred):
		y_pred, y_true = np.array(y_pred), np.array(y_true)
		assert y_true.shape == y_pred.shape
		return np.sum(np.logical_and(y_pred == 0, y_true == 1))

class FalsePositives(Metric):
	def __call__(self, y_true,y_pred):
		y_pred, y_true = np.array(y_pred), np.array(y_true)
		assert y_true.shape == y_pred.shape
		return np.sum(np.logical_and(y_pred == 1, y_true == 0))

class TrueNegatives(Metric):
	def __call__(self, y_true,y_pred):
		y_pred, y_true = np.array(y_pred), np.array(y_true)
		assert y_true.shape == y_pred.shape
		return np.sum(np.logical_and(y_pred == 0, y_true == 0))

class TruePositives(Metric):
	def __call__(self, y_true,y_pred):
		y_pred, y_true = np.array(y_pred), np.array(y_true)
		assert y_true.shape == y_pred.shape
		return np.sum(np.logical_and(y_pred == 1, y_true == 1))

class Precision(Metric):
	def __call__(self, y_true,y_pred):
		y_pred, y_true = np.array(y_pred), np.array(y_true)
		assert y_true.shape == y_pred.shape
		TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
		FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
		return TP/(TP+FP)

class Recall(Metric):
	def __call__(self, y_true,y_pred):
		y_pred, y_true = np.array(y_pred), np.array(y_true)
		assert y_true.shape == y_pred.shape
		TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
		FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
		return TP/(TP+FN)

class Fscore(Metric):
	def __call__(self, y_true,y_pred):
		y_pred, y_true = np.array(y_pred), np.array(y_true)
		assert y_true.shape == y_pred.shape
		TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
		FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
		FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
		return 2*TP/(2*TP+FP+FN)
class BinaryCrossentropy(Metric):
	def __call__(self, y_true,y_pred):
		y_pred, y_true = np.array(y_pred), np.array(y_true)
		assert y_true.shape == y_pred.shape
		return - np.mean(np.multiply(y_true, np.log(y_pred)) + np.multiply((1 - y_true), np.log(1 - y_pred)))

class CategoricalCrossentropy(Metric):
	def __call__(self, y_true, y_pred):
		y_pred,y_true=np.array(y_pred),np.array(y_true)
		assert y_true.shape == y_pred.shape
		m = y_true.shape[0]
		return -np.sum(y_true * np.log(y_pred + 1e-9))/m

class SparseCategoricalCrossentropy(Metric):
	def __call__(self, y_true,y_pred):
		y_pred = np.array(y_pred)
		y_true = np.array(self._convertToOneHot(y_true))
		assert y_true.shape == y_pred.shape
		m = y_true.shape[0]
		return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

	def _convertToOneHot(self, vector, num_classes=None):
		vector = np.array(vector)
		assert isinstance(vector, np.ndarray)
		assert len(vector) > 0

		if num_classes is None:
			num_classes = np.max(vector) + 1
		else:
			assert num_classes > 0
			assert num_classes >= np.max(vector)

		result = np.zeros(shape=(len(vector), num_classes))
		result[np.arange(len(vector)), vector] = 1
		return result.astype(int)

