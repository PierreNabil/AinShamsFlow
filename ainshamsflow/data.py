"""Data Module.

In this Module, we include our dataset handling classes. These include a general purpose Dataset class
and a ImageDataGenerator Class that is more specific to dealing with Images inside directories.
"""

import numpy as np
import matplotlib.image as mpimg
import os

from ainshamsflow.utils.asf_errors import UnsupportedShapeError, UninitializedDatasetError

__pdoc__ = dict()

__pdoc__['Dataset.__bool__'] = True
__pdoc__['Dataset.__len__'] = True
__pdoc__['Dataset.__iter__'] = True
__pdoc__['Dataset.__next__'] = True


class Dataset:
	def __init__(self, x=None, y=None):
		self.data = None
		self.target = None
		if x is not None:
			self.data = np.array(x)
		if y is not None:
			self.target = np.array(y)
		if x is not None and y is not None:
			if self.data.shape[0] != self.target.shape[0]:
				raise UnsupportedShapeError(x.shape[0], y.shape[0])
		self.is_batched = False

	def __bool__(self):
		"""Returns True if the Dataset is not None."""
		return self.data is not None

	def __len__(self):
		"""Returns the number of data points in the dataset."""
		return self.cardinality()

	def __iter__(self):
		"""Defines the Dataset as an Interator.

		Enables the user to do this:
		```python
		>>> for x, y in ds.batch(1):
		... 	print(x, y)
		```
		"""
		if self.data is None:
			raise UninitializedDatasetError

		if not self.is_batched:
			self.batch(self.cardinality())

		self.index = 0
		return self

	def __next__(self):
		"""Defines the Dataset as an Interator.

		Enables the user to do this:
		```python
		>>> for x, y in ds.batch(1):
		... 	print(x, y)
		```
		"""
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
		""" Returns a copy of the dataset."""
		dataset_copy = Dataset()
		dataset_copy.data = np.copy(self.data)
		dataset_copy.target = np.copy(self.target)
		return dataset_copy

	def apply(self, transformation):
		"""Applies a transformation function to this dataset.
		Args:
			   transformation: a function that applies to the dataset as a whole.
		Returns:
			 Dataset After transformation
		"""
		return transformation(self)

	def batch(self, batch_size):
		""" Divides the dataset into equal parts of size equals batch_size.

		If the dataset is already batched, this functions unbaches first, then baches the data
		again with the new batch_size.

		Args:
			batch_size: the size of the batches
		"""
		if self.is_batched:
			self.unbatch()

		if self.data is None:
			raise UninitializedDatasetError

		m = self.data.shape[0]
		remainder = m % batch_size

		self.take(m-remainder)

		_, *nd = self.data.shape
		_, *nt = self.target.shape
		self.data = self.data.reshape((-1, batch_size, *nd))
		self.target = self.target.reshape((-1, batch_size, *nt))

		self.is_batched = True
		return self

	def unbatch(self):
		"""Unbatches the Dataset if batched"""
		if self.is_batched:
			n1d, n2d, *nd = self.data.shape
			self.data = self.data.reshape((n1d*n2d, *nd))
			if self.target is not None:
				n1t, n2t, *nt = self.target.shape
				self.target = self.target.reshape((n1t*n2t, *nt))
		return self

	def cardinality(self):
		""" Returns the number of data points in the dataset."""
		if self.data is None:
			raise UninitializedDatasetError

		if self.is_batched:
			return self.data.shape[0] * self.data.shape[1]
		else:
			return self.data.shape[0]

	def concatenate(self, ds_list):
		"""Concatenates this Dataset with any number of datasets.

		Args:
			ds_list: a list of datasets to use for concatenation.
		"""

		if self.data is None:
			raise UninitializedDatasetError
		for ds in ds_list:
			if ds.data is None:
				raise UninitializedDatasetError

		self.data = np.concatenate((self.data, *[ds.data for ds in ds_list]))
		if self.target is not None:
			self.target = np.concatenate((self.target, *[ds.target for ds in ds_list]))
		return self

	def enumerate(self):
		"""Enumerates the dataset into numbered data elements.

		enables the user to do this:
		```python
		>>> ds.enumerate().batch(1)
		>>> for (i, x), y in ds:
		... 	print(i, x, y)
		```
		"""
		if self.data is None:
			raise UninitializedDatasetError
		enum = []
		for i in range(self.cardinality()):
			enum.append([i, self.data[i]])
		self.data = np.array(enum)
		return self

	def filter(self, function):
		"""Filters the dataset given a certain function.

		Args:
			function: specifies a condition [function(x)] to keep each element x in the dataset.
		"""
		if self.data is None:
			raise UninitializedDatasetError
		new_data = []
		for x in self.data:
			if function(x):
				new_data.append(x)
		self.data = np.array(new_data)
		return self

	def numpy(self):
		"""Returns the Dataset in a numpy format.

		Returns:
			dataset features: if dataset is not labeled
			dataset features, dataset labels: if dataset is labeled
		"""
		if self.data is None and self.target is None:
			raise UninitializedDatasetError
		elif self.target is None:
			return self.data
		elif self.data is None:
			return self.target
		else:
			return self.data, self.target

	def map(self, function):
		"""Maps each element in the dataset with the function map."""
		if self.data is None:
			raise UninitializedDatasetError
		function = np.vectorize(function)
		return function(self.data)

	def range(self, *args):
		"""Returns a new Dataset with data as a numbered list.

		uses the same syntax as np.arange()
		"""
		self.data = np.arange(*args)
		return self

	def shuffle(self):
		"""Arrays shuffled in-place by their first dimension"""

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
		"""Splits the dataset into 2 batches (training and testing/validation)

			Args:
				split_percentage: (float) percentage of the testing/validation data points
				shuffle: (bool)	if true, the data is shuffled before the split

			Returns:
				If the dataset was initialized with x only:	returns unlabeled ds_train, ds_test
				If the dataset was initialized with x and y: returns labeled ds_train, ds_test
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
		"""Takes the first N of data points in dataset to a given end.

		usefull for visulaizing the dataset as follows:
		```python
		>>> for x, y in ds.copy().take(5):
		... 	print(x, y)
		```
		"""
		if self.data is None and self.target is None:
			raise UninitializedDatasetError
		if self.data is not None:
			self.data = self.data[:limit]
		if self.target is not None:
			self.target = self.target[:limit]
		return self

	def skip(self, limit):
		"""a function that skips data in dataset from a given start."""
		if self.data is None and self.target is None:
			raise UninitializedDatasetError
		if self.data is not None:
			self.data = self.data[limit:]
		if self.target is not None:
			self.target = self.target[limit:]
		return self

	def add_data(self, x):
		"""Add features to a dataset."""
		x = np.array(x)
		if self.target is not None:
			if x.shape[0] != self.target.shape[0]:
				raise UnsupportedShapeError(x.shape[0], self.target.shape[0])
		self.data = x
		return self

	def add_targets(self, y):
		"""add labels to a dataset."""
		y = np.array(y)
		if self.data is not None:
			if self.data.shape[0] != y.shape[0]:
				raise UnsupportedShapeError(y.shape[0], self.data.shape[0])
		self.target = y

		return self

	def normalize(self):
		"""normalize the dataset so as to be centred around the origin."""
		if self.data is None:
			raise UninitializedDatasetError
		self.data = (self.data-self.data.mean(axis=0))/np.sqrt(self.data.var(axis=0)+1e-6)
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

		Args:
			directory: If directory holds images only, this function will use these images as a dataset without any labels.
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
		"""Defines the ImageDataGenerator as an Interator.

		Enables the user to do this:
		```python
		>>> for x, y in ds.batch(1):
		... 	print(x, y)
		```
		"""
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
		"""Returns a copy of the dataset."""
		dataset_copy = ImageDataGenerator()
		dataset_copy.data = np.copy(self.data)
		dataset_copy.target = np.copy(self.target)
		dataset_copy.dir = self.dir
		dataset_copy.class_name = self.class_name[:]
		return dataset_copy
	
	def split(self, split_percentage, shuffle=False):
		"""Splits the dataset into 2 batches (training and testing/validation)

			Args:
				split_percentage: (float) percentage of the testing/validation data points
				shuffle: (bool)	if true, the data is shuffled before the split

			Returns:
				If the dataset was initialized with x only:	returns unlabeled ds_train, ds_test
				If the dataset was initialized with x and y: returns labeled ds_train, ds_test
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
