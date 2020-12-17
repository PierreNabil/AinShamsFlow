

class ASFError(Exception):
	pass


class UncompiledModelError(ASFError):
	def __str__(self):
		return 'Must compile model before use.'


class MultipleAccessError(ASFError, IndexError):
	def __str__(self):
		return 'Trying to access layers using 2 methods simultaneously.'


class BaseClassError(ASFError, NotImplementedError):
	def __str__(self):
		return 'Trying to use a Base Class which is not meant for use.'


class UnsupportedShapeError(ASFError):
	def __init__(self, given_shape, expected_shape):
		self.given_shape = str(given_shape)
		self.expected_shape = str(expected_shape)

	def __str__(self):
		return 'Expected: {}, Found: {}.'.format(self.expected_shape, self.given_shape)


class NameNotFoundError(ASFError):
	def __init__(self, name, module_name):
		self.name = name
		self.module_name = module_name

	def __str__(self):
		return 'Name {} not found in {}.'.format(self.name, self.module_name)