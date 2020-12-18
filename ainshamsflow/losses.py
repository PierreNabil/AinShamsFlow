import numpy as np
from peras_errors import BaseClassError


""" Written by Ziad Tarek """


class Loss:
    def __call__(self, y_pred, y_true):
        raise BaseClassError

    def diff(self, y_pred, y_true):
        raise BaseClassError


class MSE(Loss):
    name = 'Mean Square error '

    def __call__(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape
        m = np.sum(y_true.shape[1])
        return np.sum(np.square(y_pred - y_true), axis=1, keepdims=True) / (2 * m)

    def diff(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape
        m = np.sum(y_true.shape[1])
        return np.sum(y_pred - y_true, axis=1, keepdims=True) / m


class MAE(Loss):
    name = 'Mean Absolute Error'

    def __call__(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape
        m = np.sum(y_true.shape[1])
        return np.sum(np.abs(y_pred - y_true), axis=1, keepdims=True) / m

    def diff(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape
        m = np.sum(y_true.shape[1])
        return np.where(y_pred > y_true, 1, -1) / m


class HuberLoss(Loss):
    name = "huber loss error"

    def __init__(self, delta):
        self.delta = delta

    def __call__(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape
        super(HuberLoss, self).__call__(y_true, y_pred)
        if np.abs(y_pred - y_true) <= self.delta:
            return 0.5 * np.square(y_pred - y_true)
        else:
            return (self.delta * np.abs(y_pred - y_true)) - 0.5 * np.square(self.delta)

    def diff(self, y_pred, y_true):
        super(HuberLoss, self).diff(y_pred, y_true)
        assert y_true.shape == y_pred.shape
        if np.abs(y_pred - y_true) <= self.delta:
            return y_pred - y_true
        else:
            return self.delta * np.sign(y_pred - y_true)


class MAPE(Loss):
    __name__ = 'Mean absolute percentage error'

    def __call__(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape
        m = y_true.shape[1]
        return 1 - np.sum(np.abs(y_pred - y_true) / y_true) / m

    def diff(self, y_pred, y_true):
        # Todo:
        pass


class LogLossIdentityActivation(Loss):
    def __call__(self, y_true, y_pred):
        super(LogLossIdentityActivation, self).__call__(y_true, y_pred)
        assert y_true.shape == y_pred.shape
        return np.sum(np.log(1 + np.exp(-y_true * y_pred)))

    def diff(self, y_pred, y_true, x=None):
        assert y_true.shape == y_pred.shape
        return -(y_true * x) / (1 + np.exp(-y_true * y_pred))


class LogLossSigmoidActivation(Loss):
    def __call__(self, y_true, y_pred):
        super.__call__(y_true, y_pred)
        assert y_true.shape == y_pred.shape
        return -np.log(np.abs(y_true / 2 - 1 / 2 + y_pred))

    def diff(self, y_pred, y_true, w=None, x=None):
        return -x / (y_true / 2 - 0.5 + y_pred)


class MultiClassificationsLoss(Loss):
    def __call__(self, y_pred):
        return -np.log(y_pred)


class PerceptronCriterionLoss(Loss):
    def __call__(self, y_true, y_pred):
        super.__call__(y_true, y_pred)
        assert y_true.shape == y_pred.shape
        return np.max(0, -y_true * y_pred)

    def diff(self, y_pred, y_true, w=None, x=None):
        if y_true * w * x <= 0:
            return -y_true * x
        else:
            return 0


class NonDifferentiableLoss(Loss):
    def __call__(self, y_true, y_pred, w=None, x=None):
        super.__call__(y_true, y_pred)
        y_pred = w * x
        assert y_true.shape == y_pred.shape
        return np.square(y_true - np.sign(y_pred))
    # No derivative as sign is non-differentiable


class SSDLoss(Loss):
    def __call__(self, y_true, y_pred):
        super().__call__(y_true, y_pred)
        return np.square(y_true - y_pred)

    def diff(self, y_pred, y_true, w=None, x=None):
        assert y_true.shape == y_pred.shape
        return -2 * x * (y_true - w * x)


class SvmHingeLoss(Loss):
    def __call__(self, y_true, y_pred):
        super().__call__(y_true, y_pred)
        return np.max(0, 1 - y_true * y_pred)

    def diff(self, y_pred, y_true, w=None, x=None):
        if y_true * w * x <= 1:
            return -y_true * x
        else:
            return 0
