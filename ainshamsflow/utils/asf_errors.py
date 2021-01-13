"""Exceptions Module for asf.

In this Module, we define all Exceptions used in te asf
framework.
"""


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


class LayerNotFoundError(ASFError):
	def __init__(self, name_or_id, name):
		self.is_name = False if name_or_id == 'id' else True
		self.name = name

	def __str__(self):
		if self.is_name:
			return 'Layer {} not found.'.format(self.name)
		else:
			return 'Layer indexed {} not found.'.format(self.name)


class UninitializedDatasetError(ASFError):
	def __str__(self):
		return 'Using an uninitialized dataset is invalid.'
