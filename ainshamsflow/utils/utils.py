from ainshamsflow.data import Dataset

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
