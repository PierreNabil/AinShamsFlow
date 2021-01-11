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
	def apply(self):
		pass

	def numpy(self):
		pass

	def batch(self,batch_size,drop_reminder=True):
		i = 0
		dt = list()
		targ - list()
		sz = len(self.data)
		if drop_reminder==True:
			sz-=sz%batch_size
		while i < sz:
			t=i
			i+=batch_size
			dt.append(self.data[t:i])
			targ.append(self.target[t:i])
		self.data=np.array(dt)
		self.target=np.array(targ)
        
	def cardinality(self):
		pass 

	def concatenate(self):
		pass

	def enumerate(self):
		pass

	def filter(self):
		pass

	def map(self,func):
		ds=[]
		for element in self.data:
			ds.append(func(element))
		self.data=np.array(ds)


	def range(self):
		pass

	def reduce(self):
		pass

	def shuffle(self):
		pass

	def skip(self):
		pass

	def take(self,limit):
		return self.data[:limit]
		

	def unbatch(self):
		pass

	def zip(self):
		pass
	
	def normalize(self):
		pass


class ImageDataGenerator(Dataset):
	def __init__(self):
		pass

	def flow_from_directory(self, directory):
		pass
