import numpy as np
from peras_errors import BaseClassError, NameNotFoundError

""" Written by Ziad Tarek """


def get(loss_name):
    """Get any Loss in this module by name"""
    losses = [MSE, MAE, MAPE, HuberLoss, LogLossLinearActivation, LogLossSigmoidActivation, MultiClassificationsLoss,
              PerceptronCriterionLoss, SSDLoss, SvmHingeLoss, BinaryCrossEntropy, CategoricalCrossEntropy]
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
        # raise BaseClassError
        pass

    def diff(self, y_pred, y_true):
        # raise BaseClassError
        pass


class MSE(Loss):
    """Mean squared error loss function"""
    __name__ = 'MSE'

    def __call__(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        m = np.sum(y_true.shape[1])
        return np.sum(np.square(y_pred - y_true), axis=1, keepdims=True) / (2 * m)

    def diff(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        m = np.sum(y_true.shape[1])
        return np.sum(y_pred - y_true, axis=1, keepdims=True) / m


class MAE(Loss):
    """Mean Absolute Error"""
    __name__ = 'MAE'

    def __call__(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        m = np.sum(y_true.shape[1])
        return np.sum(np.abs(y_pred - y_true), axis=1, keepdims=True) / m

    def diff(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        m = np.sum(y_true.shape[1])
        return np.where(y_pred > y_true, 1, -1) / m


class HuberLoss(Loss):
    """huber loss error"""
    __name__ = 'HuberLoss'

    def __init__(self, delta):
        self.delta = delta

    def __call__(self, y_true, y_pred):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        super(HuberLoss, self).__call__(y_true, y_pred)
        if np.abs(y_pred - y_true) <= self.delta:
            return 0.5 * np.square(y_pred - y_true)
        else:
            return (self.delta * np.abs(y_pred - y_true)) - 0.5 * np.square(self.delta)

    def diff(self, y_pred, y_true):
        super(HuberLoss, self).diff(y_pred, y_true)
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        if np.abs(y_pred - y_true) <= self.delta:
            return y_pred - y_true
        else:
            return self.delta * np.sign(y_pred - y_true)


class MAPE(Loss):
    """Mean absolute percentage error"""
    __name__ = 'MAPE'

    def __call__(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        m = y_true.shape[1]
        return 1 - np.sum(np.abs(y_pred - y_true) / y_true) / m

    def diff(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        m = y_true.shape[1]
        return np.where(y_pred > y_true, 1 / (m * y_true), -1 / (m * y_true))


class LogLossLinearActivation(Loss):
    """Logistic loss in case of identity(linear) activation function"""
    __name__ = "LogLossLinearActivation"

    def __call__(self, y_true, y_pred):
        super(LogLossLinearActivation, self).__call__(y_true, y_pred)
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        return np.sum(np.log(1 + np.exp(-y_true * y_pred)))

    def diff(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        return -y_true / (1 + np.exp(-y_true * y_pred))


class LogLossSigmoidActivation(Loss):
    __name__ = "LogLossSigmoidActivation"

    def __call__(self, y_true, y_pred):
        super.__call__(y_true, y_pred)
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        return -np.log(np.abs(y_true / 2 - 1 / 2 + y_pred))

    def diff(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        return -1 / (y_true / 2 - 0.5 + y_pred)


class MultiClassificationsLoss(Loss):
    __name__ = 'MultiClassificationsLoss'

    def __call__(self, y_true, y_pred):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        return -np.sum(np.log(y_true * y_pred), axis=1)


class PerceptronCriterionLoss(Loss):
    __name__ = 'PerceptronCriterionLoss'

    def __call__(self, y_true, y_pred):
        super.__call__(y_true, y_pred)
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        return np.max(0, -y_true * y_pred)

    def diff(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        return np.where(y_true * y_pred <= 0, -y_true, 0)
        # if y_true * y_pred <= 0:  # if y_true * w * x <= 0:
        #     return -y_true * x
        # else:
        #     return 0


class SSDLoss(Loss):
    __name__ = 'SSDLoss'

    def __call__(self, y_true, y_pred):
        super().__call__(y_true, y_pred)
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        return np.square(y_true - y_pred)

    def diff(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        return -2 * (y_true - y_pred)


class SvmHingeLoss(Loss):
    __name__ = 'SvmHingeLoss'

    def __call__(self, y_true, y_pred):
        super().__call__(y_true, y_pred)
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        return np.max(0, 1 - y_true * y_pred)

    def diff(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        return np.where(y_true * y_pred <= 1, -y_true, 0)
        # if y_true * y_pred <= 1:  # if y_true * w * x <= 1:
        #     return -y_true
        # else:
        #     return 0


class BinaryCrossEntropy(Loss):
    __name__ = 'BinaryCrossEntropy'

    def __call__(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        m = np.sum(y_true.shape[1])
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m

    def diff(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        return -((y_true / y_pred) - (1 - y_true / 1 - y_pred))


class CategoricalCrossEntropy(Loss):
    __name__ = 'CategoricalCrossEntropy'

    def __call__(self, y_true, y_pred):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        m = np.sum(y_true.shape[1])
        return -np.sum(y_true * np.log(y_pred)) / m

    def diff(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        m = np.sum(y_true.shape[1])
        return -(y_true / y_pred) / m


# To be implemented
class SparseCategoricalCrossentropy(Loss):
    def __call__(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        pass

    def diff(self, y_pred, y_true):
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_true.shape == y_pred.shape
        pass
