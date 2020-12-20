import numpy as np


class Dataset:
	def __init__(self, iterator):
		#dimentions: (batch, m, n)
		self.data = np.array(iterator).T

	def __str__(self):
		return str(self.data.T)

	def __bool__(self):
		pass

	def __len__(self):
		pass

	def __iter__(self):
		self.index = 0
		return self

	def __next__(self):
		if self.index > np.data.shape[0]:
			raise StopIteration
		batch = np.data[self.index].T
		self.index += 1
		return batch

	def apply(self):
		pass

	def numpy(self):
		pass

	def batch(self):
		pass

	def cardinality(self):
		pass

	def concatenate(self):
		pass

	def enumerate(self):
		pass

	def filter(self):
		pass

	def map(self):
		pass

	def range(self):
		pass

	def reduce(self):
		pass

	def shuffle(self):
		pass

	def skip(self):
		pass

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
