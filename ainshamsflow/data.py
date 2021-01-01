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
		if self.index < self.data.shape[0]:
			x = self.data[self.index]
			y = self.target[self.index]
			self.index += 1
			return f"X: {x}\tY: {y}"
		else:
			raise StopIteration

	def apply(self, transformation):
		pass

	def numpy(self):
		pass

	def batch(self):
		pass

	def cardinality(self):
		assert(self.data is not None)
		return self.data.shape[0]

	def concatenate(self):
		pass

	def enumerate(self):
		pass

	def filter(self):
		pass

	def map(self):
		pass

	def range(self, stop, start=None, step=None):
		if start and step:
			self.__init__(x=[i for i in range(start, stop, step)])
		elif start:
			self.__init__(x=[i for i in range(start, stop)])
		else:
			self.__init__(x=[i for i in range(stop)])

	def reduce(self):
		pass

	def shuffle(self):
		""" Arrays shuffled in-place by their first dimension - nothing returned. """

		# Generate random seed
		seed = np.random.randint(0, 2 ** (32 - 1) - 1)

		if self.target is not None:
			# Ensure self.data and self.target have the same length along their first dimension.
			assert(self.data.shape[0] == self.target.shape[0])

			# Shuffle both arrays in-place using the same seed
			for array in [self.data, self.target]:
				# Generate random state object
				r_state = np.random.RandomState(seed)
				r_state.shuffle(array)

		else:
			# Generate random state object
			r_state = np.random.RandomState(seed)

			# Shuffle the data array only
			r_state.shuffle(self.data)

	def skip(self):
		pass

	def split(self, split_percentage, shuffle=False):

		"""
			Splits the dataset into 2 batches (training and testing/validation)

				Inputs:
					- split_percentage: (float) percentage of the testing/validation data points
					- shuffle: 			(bool)	if true, the data is shuffle before the split

				Returns:
					- If the dataset was initialized with x (self.data) only: returns x_train, x_test
					- If the dataset was initialized with x and y:			  returns x_train, y_train, x_test, y_test
		"""

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


x = np.random.randint(0, 10, (10, 5))
y = np.random.randint(0, 10, (10, 1))

ds = Dataset(x=x, y=y)


for data_point in ds:
	print(data_point)
"""
X: [2 1 7 9 7]	Y: [2]
X: [9 0 2 9 0]	Y: [4]
X: [4 4 0 9 0]	Y: [6]
X: [8 6 6 8 3]	Y: [9]
X: [8 8 8 2 9]	Y: [3]
X: [5 8 1 5 5]	Y: [6]
X: [4 4 6 8 1]	Y: [5]
X: [1 9 8 0 3]	Y: [8]
X: [8 6 7 2 8]	Y: [8]
X: [1 8 9 7 2]	Y: [6]
"""

# ds.shuffle()
"""
x_train, y_train, x_test, y_test = ds.split(0.3, shuffle=True)
print(x_test)
print(y_test)
"""