"""Utils module for miscellaneous helper functions."""

import numpy as np

from ainshamsflow.data import Dataset


def pred_one_hot(y_pred):
	n_c = y_pred.shape[-1]
	return np.squeeze(np.eye(n_c)[np.argmax(y_pred, axis=-1)])


def true_one_hot(y_true, n_c):
	m = y_true.shape[0]
	return np.squeeze(np.eye(n_c)[y_true]).reshape((m, -1))


def get_dataset_from_xy(x, y):
	if x is None:
		raise RunningWithoutDataError

	elif isinstance(x, Dataset):
		if x.data is None:
			raise RunningWithoutDataError

		elif x.target is None:
			if y is None:
				raise RunningWithoutDataError
			elif isinstance(y, Dataset):
				return x.add_targets(y.target)
			else: # isinstance(y, np.array)
				return x.add_targets(y)

		else:  # x.target is not None
			return x

	else:  # isinstance(x, np.array)
		if y is None:
			raise RunningWithoutDataError
		elif isinstance(y, Dataset):
			return y.add_data(x)
		else: # isinstance(y, np.array)
			return Dataset(x, y)
