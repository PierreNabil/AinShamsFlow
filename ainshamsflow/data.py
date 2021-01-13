import numpy as np


class Dataset:
	def __init__(self, x=None, y=None):
		if x is not None:
			self.data = np.array(x)
		else:
			self.data = None

		if y is not None:
			self.target = np.array(y)
		else:
			self.target = None

	def __bool__(self):
		return self.data is not None and self.target is not None

	def __len__(self):
		return self.cardinality()

	def __iter__(self):
		self.index = 0
		return self

	def __next__(self):
		assert self.data is not None
		if self.index >= self.data.shape[0]:
			raise StopIteration

		if self.data is not None and self.target is not None:
			x = self.data[self.index]
			y = self.target[self.index]
			self.index += 1
			return x, y

		elif self.data is not None:
			x = self.data[self.index]
			self.index += 1
			return x

	def copy(self):
		""" Returns a copy of the dataset """
		dataset_copy = Dataset()
		dataset_copy.data = self.data
		dataset_copy.target = self.target
		return dataset_copy

	def apply(self, transformation):
		return transformation(self)

	def batch(self, batch_size, drop_remainder=False):
		
		""" 
			Divides the dataset into equal parts.

			Inputs:
				- batch_size:		(int)
				- drop_remainder:	(bool)

			Returns:
				- (list of ndarrays) dividing the self.data into sub-arrays
		"""

		if drop_remainder:
			return np.split(self.data[:-(self.cardinality() % batch_size)], batch_size)
		else:
			batches = np.split(self.data[:-(self.cardinality() % batch_size)], batch_size)
			remainder = np.array(self.data[-(self.cardinality() % batch_size):])
			batches.append(remainder)
			return batches

	def cardinality(self):
		""" Returns the number of data points in the dataset """
		assert self.data is not None
		return self.data.shape[0]

	def concatenate(self, ds_list):
		"""
			Creates a Dataset by concatenating the given dataset with this dataset.

			Inputs:
				- ds_list: (list) of the datasets to be concatenated

			Returns:
				- A new concatenated dataset.
		"""
		return Dataset(x=np.concatenate((self.data, *[ds.data for ds in ds_list])))

	def enumerate(self):
		enum = []
		for i in range(self.cardinality()):
			enum.append([i, self.data[i]])
		return np.array(enum)

	def filter(self, function):
		new_data = []
		for x in self.data:
			if function(x):
				new_data.append(x)
		self.data = np.array(new_data)
		return self

	def map(self, function):
		new_data = []
		for element in self.data:
			new_data.append(function(element))
		self.data = np.array(new_data)
		return self

	def range(self, *args):
		self.data = np.arange(*args)
		return self

	def shuffle(self):
		""" Arrays shuffled in-place by their first dimension - self returned """

		assert self.data is not None

		# Generate random seed
		seed = np.random.randint(0, 2 ** (32 - 1) - 1)

		if self.target is not None:
			# Ensure self.data and self.target have the same length along their first dimension
			assert self.data.shape[0] == self.target.shape[0]

			# Shuffle both arrays in-place using the same seed
			for array in [self.data, self.target]:
				# Generate random state object
				r_state = np.random.RandomState(seed)
				r_state.shuffle(array)

		else:
			# Generate random state object and only shuffle the data array
			r_state = np.random.RandomState(seed)
			r_state.shuffle(self.data)

		return self

	def split(self, split_percentage, shuffle=False):

		"""
		Splits the dataset into 2 batches (training and testing/validation)

			Inputs:
				- split_percentage: (float) percentage of the testing/validation data points
				- shuffle:			(bool)	if true, the data is shuffled before the split

			Returns (as numpy arrays):
				- If the dataset was initialized with x only:	returns x_train, x_test
				- If the dataset was initialized with x and y:	returns x_train, y_train, x_test, y_test
		"""

		assert self.data is not None
		holdout = int(split_percentage * self.data.shape[0])
		if shuffle:
			self.shuffle()

		x_test = self.data[:holdout]
		x_train = self.data[holdout:]

		if self.target is not None:
			y_test = self.target[:holdout]
			y_train = self.target[holdout:]
			return x_train, y_train, x_test, y_test
		return x_train, x_test

	def take(self, limit):
		return self.data[:limit]


class ImageDataGenerator(Dataset):
	def __init__(self):
		pass

	def flow_from_directory(self, directory):
		pass


if __name__ == '__main__':

	# Create a dataset object
	ds = Dataset()

	# Range
	ds.range(5, 10, 2)
	for x in ds:
		print(x)

	# Cardinality
	print(ds.cardinality())

	# Initialize with lists
	x = [[10, 10, 10], [20, 20, 20], [30, 30, 30], [40, 40, 40]]
	y = [1, 2, 3, 4]

	ds = Dataset(x, y)
	for x, y in ds:
		print(x, y)

	# Shuffle
	print()
	ds.shuffle()
	for x, y in ds:
		print(x, y)

	# Split
	x = np.random.randint(0, 9, (10, 3))
	y = np.random.randint(0, 2, (10, 1))
	ds = Dataset(x, y)
	x_train, y_train, x_test, y_test = ds.split(split_percentage=0.3, shuffle=False)

	# Copy
	ds_copy = ds.copy()

	# Filter
	ds = Dataset()
	ds.range(10)

	def filter_function(x):
		return x > 5

	print(ds.data)
	ds.filter(filter_function)
	print(ds.data)

	# Map
	def map_function(x):
		return x + 10
	ds.map(map_function)
	print(ds.data)

	# Take
	print(ds.take(2))