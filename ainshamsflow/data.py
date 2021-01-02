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

	def __str__(self):
		pass

	def __bool__(self):
		pass

	def __len__(self):
		pass

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

	def apply(self, transformation):
		pass

	def numpy(self):
		pass

	def batch(self):
		pass

	def cardinality(self):
		""" Returns the number of data points in the dataset """
		assert self.data is not None
		return self.data.shape[0]

	def concatenate(self):
		pass

	def enumerate(self):
		pass

	def filter(self):
		pass

	def map(self):
		pass

	def range(self, *args):
		self.data = np.arange(*args)

	def reduce(self):
		pass

	def shuffle(self):
		""" Arrays shuffled in-place by their first dimension - nothing returned """

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

	def skip(self):
		pass

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

	def take(self):
		pass

	def unbatch(self):
		pass

	def zip(self):
		pass


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
	ds.shuffle()
	for x, y in ds:
		print(x, y)

	# Split
	x = np.random.randint(0, 9, (10, 3))
	y = np.random.randint(0, 2, (10, 1))
	ds = Dataset(x, y)
	x_train, y_train, x_test, y_test = ds.split(split_percentage=0.3, shuffle=False)
	print(x_train.shape)
	print(y_train.shape)
	print(x_test.shape)
	print(y_test.shape)
