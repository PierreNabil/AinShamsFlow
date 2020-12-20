import numpy as np

from ainshamsflow.utils.peras_errors import BaseClassError
#TODO: Add More Activations


class Activation:
	def __call__(self, z):
		raise BaseClassError

	def diff(self, z):
		raise BaseClassError


class Linear(Activation):
	def __call__(self, z):
		return z

	def diff(self, z):
		return 1


class Sigmoid(Activation):
	def __call__(self, z):
		return 1 / (1 + np.exp(- z))

	def diff(self, z):
		sig = self.__call__(z)
		return sig * (1 - sig)


class Tanh(Activation):
	def __call__(self, z):
		exp_pz = np.exp(z)
		exp_nz = np.exp(-z)
		return (exp_pz - exp_nz) / (exp_pz + exp_nz)

	def diff(self, z):
		tanh = self.__call__(z)
		return 1 - np.square(tanh)


class ReLU(Activation):
	def __call__(self, z):
		return np.maximum(0, z)

	def diff(self, z):
		return np.where(z > 0, 1, 0)


class LeakyReLU(Activation):
	def __init__(self, alpha=0.01):
		self.alpha = alpha

	def __call__(self, z):
		return np.maximum(self.alpha * z, z)

	def diff(self, z):
		return np.where(z > 0, 1, self.alpha)


#todo : hard simgoid , selu , elu , soft sign , swish , softplus , hard tanh
class Softplus(Activation):
	def __call__(self,z):
		return np.log( 1 + np.exp(z))
	def diff(self,z):
		return 1 / (1 + np.exp(-z))	# exp z / 1 + exp z

class ELU(Activation):
	def __init__(self,alpha_ELU=1.67326):
		self.alpha_ELU = alpha_ELU
	
	def __call__(self,z):
		return np.where(z <= 0 ,self.alpha_ELU * ( np.exp(z) - 1), z)
	
	def diff(self,z):
		return np.where(z < 0 , self.alpha_ELU * np.exp(z) , 1) # alpha = ????

class SELU(Activation):
	def __init__(self ,alpha_SELU = 1.67326 , lambda_SELU =1.0507):
		self.alpha_SELU = alpha_SELU
		self.lambda_SELU = lambda_SELU
	
	def __call__(self,z):
		return self.lambda_SELU * (np.where(z >= 0 , z ,self.alpha_SELU * ( np.exp(z) - 1) ))

	def diff(self,z):
		return self.lambda_SELU * (np.where(z >= 0 , 1 ,self.alpha_SELU * ( np.exp(z) )))

class Softsign(Activation):
	def __call__(self,z):
		return z / ( 1 + np.abs(z))
	
	def diff(self,z):
		Softsign_down = ( 1 + np.abs(z))
		return 1 / np.power(Softsign_down  , 2)

class Swish(Activation):
	def __call__(self,z):
		return z * Sigmoid(z)
	
	def diff(self,z):
		return Sigmoid(z) + z * Sigmoid(z) * (1 - Sigmoid(z)) 

class Softmax(Activation):
	def __call__(self,z):
		exp_z = np.exp(z)
		sum = np.sum(exp_z,axis=-1,keepdims=True)
		return exp_z / sum

	def diff(self,z):
		s = self.__call__(z)
		return np.diagflat(s) - np.dot( s , np.transpose(s))

class Hardtanh(Activation):
	def __call__(self,z):
		z = np.where(z >  1,  1, z)
		z = np.where(z < -1, -1, z)
		return z

	def diff(self,z):
		return np.where(np.logical_not(np.logical_or(np.greater(z,1),np.less(z,-1))) , 1 , 0)		

class Hardsigmoid(Activation):
	def __call__(self,z):
		return np.maximum(0, np.minimum(1, (z + 1) / 2))

	def diff(self,z):
		return np.where(np.logical_not(np.logical_or( np.less(z, -2.5) , np.greater(z , 2.5) ), 0.2 , 0))		

