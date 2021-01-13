"""Data Module.

In this Module, we include our dataset handling classes. These include a general purpose Dataset class
and a ImageDataGenerator Class that is more specific to dealing with Images inside directories.
"""

import numpy as np
import matplotlib.image as mpimg
import os

from ainshamsflow.utils.asf_errors import UnsupportedShapeError, UninitializedDatasetError


class Dataset:
	def __init__(self, x=None, y=None):
		self.data = None
		self.target = None
		if x is not None:
			self.data = np.array(x)
		if y is not None:
			self.target = np.array(y)
		if x is not None and y is not None:
			if x.shape[0] != y.shape[0]:
				raise UnsupportedShapeError(x.shape[0], y.shape[0])

	def __bool__(self):
		return self.data is not None

	def __len__(self):
		return self.cardinality()

	def __iter__(self):
		if self.data is None:
			raise UninitializedDatasetError
		self.index = 0
		return self

	def __next__(self):
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
		dataset_copy.data = np.copy(self.data)
		dataset_copy.target = np.copy(self.target)
		return dataset_copy

	def apply(self, transformation):
		return transformation(self)

	def batch(self, batch_size, drop_remainder=False):
		
		""" 
			Divides the dataset into equal parts of size equals batch_size.
			(Modifies self.data and self.target to be a list of arrays rather than numpy arrays)

			Inputs:
				- batch_size:		(int)
				- drop_remainder:	(bool)

			Returns:
				- self
		"""
		if self.data is None:
			raise UninitializedDatasetError

		m = self.cardinality()

		if drop_remainder:
			self.data = np.split(self.data[:-(m % batch_size)], batch_size)
			self.target = np.split(self.target[:-(m % batch_size)], batch_size)
		else:
			data_batches = np.split(self.data[:-(m % batch_size)], batch_size)
			data_remainder = self.data[-(m % batch_size):]
			data_batches.append(data_remainder)

			target_batches = np.split(self.target[:-(m % batch_size)], batch_size)
			target_remainder = self.target[-(m % batch_size):]
			target_batches.append(target_remainder)

			self.data = data_batches
			self.target = target_batches
		return self

	def cardinality(self):
		""" Returns the number of data points in the dataset """
		if self.data is None:
			raise UninitializedDatasetError
		return self.data.shape[0]

	def concatenate(self, ds_list):
		"""
			Creates a Dataset by concatenating the given dataset with this dataset.

			Inputs:
				- ds_list: (list) of the datasets to be concatenated

			Returns:
				- A new concatenated dataset.
		"""

		if self.data is None:
			raise UninitializedDatasetError

		self.data = np.concatenate((self.data, *[ds.data for ds in ds_list]))
		return self


	def enumerate(self):
		if self.data is None:
			raise UninitializedDatasetError
		enum = []
		for i in range(self.cardinality()):
			enum.append([i, self.data[i]])
		self.data = np.array(enum)
		return self

	def filter(self, function):
		if self.data is None:
			raise UninitializedDatasetError
		new_data = []
		for x in self.data:
			if function(x):
				new_data.append(x)
		self.data = np.array(new_data)
		return self

	def map(self, function):
		if self.data is None:
			raise UninitializedDatasetError
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

		if self.data is None:
			raise UninitializedDatasetError

		# Generate random seed
		seed = np.random.randint(0, 2 ** (32 - 1) - 1)

		if self.target is not None:
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

		if self.data is None:
			raise UninitializedDatasetError
		holdout = int(split_percentage * self.data.shape[0])
		if shuffle:
			self.shuffle()

		x_test = self.data[holdout:]
		x_train = self.data[:holdout]

		if self.target is not None:
			y_test = self.target[holdout:]
			y_train = self.target[:holdout]
		else:
			y_train = None
			y_test = None

		return Dataset(x=x_train, y=y_train), Dataset(x=x_test, y=y_test)

	def take(self, limit):
		if self.data is None:
			raise UninitializedDatasetError

		self.data = self.data[:limit]
		return self

	def skip(self, limit):
		if self.data is None:
			raise UninitializedDatasetError
		self.data = self.data[limit:]
		return self


class ImageDataGenerator(Dataset):
	"""Image Data Generator Class.

	This class helps in training large amounts of images with minimal memory allocation.
	"""
	def __init__(self):
		self.class_name = []
		self.dir = None
		super().__init__()

	def flow_from_directory(self, directory):
		"""Reads Images from a Directory.

		If directory holds images only, this function will use these images as a dataset without any labels.
		Otherwise, if the directory holds folders of images, it will store the folder names as class names in
		the class_names attribute. It will then label the images according to their folders.
		"""
		self.dir = directory
		self.class_name = [name for name in os.listdir(directory)
						   if os.path.isdir(os.path.join(directory, name))]
		if class_name:
			images = []
			labels = []
			for i in range(len(self.class_name)):
				for img_name in os.listdir(self.class_name[i]):
					if os.path.isfile(os.path.join(directory, self.class_name[i], img_name)):
						images.append(img_name)
						labels.append(i)
			self.data = np.array(images)
			self.target = np.array(labels)
		else:
			images = [img_name for img_name in os.listdir(directory)
					   if os.path.isfile(os.path.join(directory, img_name))]
			self.data = np.array(images)

		self.shuffle()
		return self

	def __next__(self):
		if self.index >= self.data.shape[0]:
			raise StopIteration
		self.index += 1
		img = self._extract_img(self.data[self.index-1], self.target[self.index-1])
		if self.target is not None:
			label = self.target[self.index-1]
			return img, label
		else:
			return img

	def _extract_img(self, filename, label):
		if isinstance(filename, str):
			ans = mpimg.imread(os.path.join(self.dir, self.class_name[label], filename))
		else:
			ans = []
			for file, lab in zip(filename, label):
				ans.append(self._extract_img(file, lab))
			ans = np.array(ans)
		return ans
	
	def copy(self):
		""" Returns a copy of the dataset """
		dataset_copy = ImageDataGenerator()
		dataset_copy.data = np.copy(self.data)
		dataset_copy.target = np.copy(self.target)
		dataset_copy.dir = self.dir
		dataset_copy.class_name = self.class_name[:]
		return dataset_copy
	
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

		if self.data is None:
			raise UninitializedDatasetError
		img_gen_train = ImageDataGenerator()
		img_gen_test = ImageDataGenerator()
		
		img_gen_train.dir = img_gen_test.dir = self.dir
		img_gen_train.class_name = self.class_name[:]
		img_gen_test.class_name = self.class_name[:]
		
		holdout = int(split_percentage * self.data.shape[0])
		if shuffle:
			self.shuffle()

		img_gen_train.data = self.data[:holdout]
		img_gen_test.data = self.data[holdout:]

		if self.target is not None:
			img_gen_train.target = self.target[:holdout]
			img_gen_test.target = self.target[holdout:]

		return img_gen_train, img_gen_test


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
	ds_train, ds_test = ds.split(split_percentage=0.3, shuffle=False)

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
	ds.take(2)

	# Batch
	#x = [[10, 10, 10], [20, 20, 20], [30, 30, 30], [40, 40, 40]]
	x = [i for i in range(5)]
	y = [1, 2, 3, 4, 5]
	ds = Dataset(x=x, y=y)

	print(ds.data)
	print(ds.target)
	ds.batch(2)
	print(ds.data)
	print(ds.target)



