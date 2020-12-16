

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
