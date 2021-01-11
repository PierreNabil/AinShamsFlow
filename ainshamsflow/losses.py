"""Losses Module.

In this Module we provide our loss functions for a variety of
use cases like Mean Squared Error or Cross Entropy loss.
"""

import numpy as np
from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError
from ainshamsflow.utils.utils import true_one_hot


def get(loss_name):
    """Get any Loss in this module by name"""
    losses = [MSE, MAE, MAPE, HuberLoss, LogLossLinear, LogLossSigmoid,
              PerceptronCriterion, SvmHingeLoss, BinaryCrossentropy, CategoricalCrossentropy,
              SparseCategoricalCrossentropy]
    for loss in losses:
        if loss.__name__.lower() == loss_name.lower():
            return loss()
    raise NameNotFoundError(loss_name, __name__)


class Loss:
    """Loss Base Class.
    To create a new Loss Function, create a class that inherits
    from this class.
    You then have to add any parameters in your constructor
    and redefine the __call__() and diff() methods.
    Note: all loss functions can be used as metrics.
    """

    def __call__(self, y_pred, y_true):
        raise BaseClassError

    def diff(self, y_pred, y_true):
        raise BaseClassError


class MSE(Loss):
    """Mean squared error loss function"""
    __name__ = 'MSE'

    def __call__(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true)) / 2

    def diff(self, y_pred, y_true):
        return np.mean(y_pred - y_true, axis=0, keepdims=True)


class MAE(Loss):
    """Mean Absolute Error"""
    __name__ = 'MAE'

    def __call__(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

    def diff(self, y_pred, y_true):
        m = y_true.shape[0]
        return np.sign(y_pred - y_true) / m


class HuberLoss(Loss):
    """Huber loss error"""
    __name__ = 'HuberLoss'

    def __init__(self, delta=1.0):
        self.delta = delta

    def __call__(self, y_true, y_pred):
        return np.where(np.abs(y_pred - y_true) <= self.delta, 0.5 * np.square(y_pred - y_true),
                        (self.delta * np.abs(y_pred - y_true)) - 0.5 * np.square(self.delta))

    def diff(self, y_pred, y_true):
        return np.where(np.abs(y_pred - y_true) <= self.delta, y_pred - y_true, self.delta * np.sign(y_pred - y_true))


class MAPE(Loss):
    """Mean absolute percentage error"""
    __name__ = 'MAPE'

    def __call__(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true) / y_true)

    def diff(self, y_pred, y_true):
        m = y_true.shape[0]
        return np.where(y_pred > y_true, 1 / (m * y_true), -1 / (m * y_true))


class LogLossLinear(Loss):
    """Logistic loss in case of identity(linear) activation function"""
    __name__ = "LogLossLinear"

    def __call__(self, y_true, y_pred):
        return np.sum(np.log(1 + np.exp(-y_true * y_pred)))

    def diff(self, y_pred, y_true):
        return -y_true * np.exp(-y_true * y_pred) / (1 + np.exp(-y_true * y_pred))


class LogLossSigmoid(Loss):
    __name__ = "LogLossSigmoid"

    def __call__(self, y_pred, y_true):
        return -np.mean(np.log(np.abs(y_true / 2 - 0.5 + y_pred)))

    def diff(self, y_pred, y_true):
        x = y_true / 2 - 0.5 + y_pred
        return -np.sign(x) / np.abs(x)


class PerceptronCriterion(Loss):
    __name__ = 'PerceptronCriterion'

    def __call__(self, y_true, y_pred):
        return np.maximum(0, -y_true * y_pred)

    def diff(self, y_pred, y_true):
        return np.where(y_true * y_pred <= 0, -y_true, 0)


class SvmHingeLoss(Loss):
    __name__ = 'SvmHingeLoss'

    def __call__(self, y_pred, y_true):
        return np.maximum(0, 1 - y_true * y_pred)

    def diff(self, y_pred, y_true):
        return np.where(y_true * y_pred <= 1, -y_true, 0)


class BinaryCrossentropy(Loss):
    __name__ = 'BinaryCrossentropy'

    def __call__(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def diff(self, y_pred, y_true):
        # m = y_true.shape[0]
        # return -np.sum(((y_true / y_pred) - (1 - y_true / 1 - y_pred))) / m
        np.mean(y_pred - y_true, axis=0, keepdims=True)


class CategoricalCrossentropy(Loss):
    __name__ = 'CategoricalCrossentropy'

    def __call__(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-6))

    def diff(self, y_pred, y_true):
        m = y_true.shape[0]
        return -(y_true / y_pred) / m


class SparseCategoricalCrossentropy(Loss):
    __name__ = 'SparseCategoricalCrossentropy'

    def __call__(self, y_pred, y_true):
        n_c = y_pred.shape[-1]
        y_true = true_one_hot(y_true, n_c)
        return -np.mean(y_true * np.log(y_pred + 1e-6))

    def diff(self, y_pred, y_true):
        n_c = y_pred.shape[-1]
        y_true = true_one_hot(y_true, n_c)
        m = y_pred.shape[0]
        return -(y_true / y_pred) / m
