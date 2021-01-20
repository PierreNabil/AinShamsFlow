"""Losses Module.

In this Module we provide our loss functions for a variety of
use cases like Mean Squared Error or Cross Entropy loss.
"""

import numpy as np
from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError
from ainshamsflow.utils.utils import true_one_hot

__pdoc__ = dict()

for loss_n in ['Loss', 'MSE', 'MAE', 'MAPE', 'HuberLoss', 'LogLossLinear', 'LogLossSigmoid',
              'PerceptronCriterion', 'SvmHingeLoss', 'BinaryCrossentropy', 'CategoricalCrossentropy',
              'SparseCategoricalCrossentropy']:
    __pdoc__[loss_n + '.__call__'] = True


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
        """Use the Loss function to get the loss Value."""
        raise BaseClassError

    def diff(self, y_pred, y_true):
        """Get the derivative of the loss function."""
        raise BaseClassError


class MSE(Loss):
    """Mean squared error loss function.

    Computes the mean squared error between labels and predictions.

    After computing the squared distance between the inputs, the mean value over
    the last dimension is returned.
    `loss = mean(square(y_true - y_pred), axis=-1)`

    Standalone usage:

    ```python
    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = asf.losses.MSE()
    >>> loss(y_pred,y_true)
    ```
    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
        error_values: Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    __name__ = 'MSE'

    def __call__(self, y_pred, y_true):
        m = y_pred.shape[0]
        return np.sum(np.square(y_pred - y_true)) / (2 * m)

    def diff(self, y_pred, y_true):
        m = y_pred.shape[0]
        return (y_pred - y_true) / m


class MAE(Loss):
    """Mean Absolute Error.

    Computes the mean absolute error between labels and predictions.

    `loss = mean(abs(y_true - y_pred), axis=-1)`

    Standalone usage:
    ```python
    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = asf.losses.MAE()
    >>> loss(y_pred,y_true)
    ```
     Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
     Returns:
        Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    __name__ = 'MAE'

    def __call__(self, y_pred, y_true):
        m = y_true.shape[0]
        return np.sum(np.abs(y_pred - y_true)) / m

    def diff(self, y_pred, y_true):
        m = y_true.shape[0]
        return np.sign(y_pred - y_true) / m


class HuberLoss(Loss):
    """Huber loss error.

    Computes the Huber loss between `y_true` and `y_pred`.

    For each value x in `error = y_true - y_pred`:
    ```
    loss = 0.5 * x^2                  if |x| <= d
    loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
    ```
    where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss
    Standalone usage:
    ```python
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> loss = asf.losses.HuberLoss()
    >>> loss(y_true, y_pred)
    0.155
    ```
    """
    __name__ = 'HuberLoss'

    def __init__(self, delta=1.0):
        self.delta = delta

    def __call__(self, y_pred, y_true):
        return np.where(np.abs(y_pred - y_true) <= self.delta, 0.5 * np.square(y_pred - y_true),
                        (self.delta * np.abs(y_pred - y_true)) - 0.5 * np.square(self.delta))

    def diff(self, y_pred, y_true):
        return np.where(np.abs(y_pred - y_true) <= self.delta, y_pred - y_true, self.delta * np.sign(y_pred - y_true))


class MAPE(Loss):
    """Mean absolute percentage error.

    Computes the mean absolute percentage error between `y_true` and `y_pred`.

    `loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)`
    Standalone usage:
    ```python
    >>> y_true = np.random.rand(2, 3)
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = asf.losses.MAPE()
    >>> loss(y_pred,y_true)
    ```
    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
        Mean absolute percentage error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    __name__ = 'MAPE'

    def __call__(self, y_pred, y_true):
        m = y_pred.shape[0]
        return np.sum(np.abs(y_pred - y_true) / y_true) / m

    def diff(self, y_pred, y_true):
        m = y_true.shape[0]
        return np.where(y_pred > y_true, 1 / (m * y_true), -1 / (m * y_true))


class LogLossLinear(Loss):
    """Logistic loss in case of identity(linear) activation function.

    Computes the logistic regression loss between y_pred and y_true.

    This class is used only in case of linear activation function and
    is provided by y_pred and y_true
    Stand alone usage:
    ```python
    >>> y_pred=2
    >>> y_true=2.4
    >>> loss=asf.losses.LogLossLinear()
    >>> loss(2.4,2)
    0.008196
    ```
    Returns:
        the logistic loss values in case of linear regression
    """
    __name__ = "LogLossLinear"

    def __call__(self, y_pred, y_true):
        return np.sum(np.log(1 + np.exp(-y_true * y_pred)))

    def diff(self, y_pred, y_true):
        return -y_true * np.exp(-y_true * y_pred) / (1 + np.exp(-y_true * y_pred))


class LogLossSigmoid(Loss):
    """Logistic loss in case of sigmoid activation functions.

       Computes the logistic regression loss between y_pred and y_true.

        This class is used only in case of linear activation function and
        is provided by y_pred and y_true
        Standalone usage:
        ```python
        >>> y_pred=2
        >>> y_true=2.4
        >>> loss=asf.losses.LogLossSigmoid()
        >>> loss(y_pred,y_true)
        ```
        Returns:
            the logistic loss values in case of linear regression
    """
    __name__ = "LogLossSigmoid."

    def __call__(self, y_pred, y_true):
        m = y_pred.shape[0]
        return -np.sum(np.log(np.abs(y_true / 2 - 0.5 + y_pred))) / m

    def diff(self, y_pred, y_true):
        x = y_true / 2 - 0.5 + y_pred
        return -np.sign(x) / np.abs(x)


class PerceptronCriterion(Loss):
    """Bipolar perceptron criterion loss class.
       Standalone usage:
       ```python
       >>> y_pred=2
       >>> y_true=2.4
       >>> loss=asf.losses.PerceptronCriterion()
       >>> loss(y_pred,y_true)
       0
       >>> loss.(-y_pred,y_true)
       4.8
       ```
       Returns:
           0 if both numbers are positives or negatives
           otherwise ,returns their product
    """
    __name__ = 'PerceptronCriterion'

    def __call__(self, y_pred, y_true):
        return np.maximum(0, -y_true * y_pred)

    def diff(self, y_pred, y_true):
        return np.where(y_true * y_pred <= 0, -y_true, 0)


class SvmHingeLoss(Loss):
    """SVM hinge criterion loss.

    Stand alone usage:
    ```python
    >>> y_pred=2
    >>> y_true=2.4
    >>> loss=asf.losses.SvmHingeLoss()
    >>> loss(y_pred,y_true)
    0
    >>> loss(-y_pred,y_true)
    5.8
    ```
     Returns:
        0 if product of the two numbers is greater than 1
        otherwise ,returns 1-their product
    """
    __name__ = 'SvmHingeLoss'

    def __call__(self, y_pred, y_true):
        return np.maximum(0, 1 - y_true * y_pred)

    def diff(self, y_pred, y_true):
        return np.where(y_true * y_pred <= 1, -y_true, 0)


class BinaryCrossentropy(Loss):
    """Binary cross entropy Loss.

    Computes the cross-entropy loss between true labels and predicted labels.

    Use this cross-entropy loss when there are only two label classes (assumed to
    be 0 and 1). For each example, there should be a single floating-point value
    per prediction.
    Standalone usage:
    ```python
    >>> y_true = [[1], [0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = asf.losses.BinaryCrossentropy()
    >>> loss(y_pred,y_true)
    0.815
    """

    __name__ = 'BinaryCrossentropy'

    def __call__(self, y_pred, y_true):
        m = y_pred.shape[0]
        return -np.sum(np.where(y_true, np.log(y_pred), np.log(1 - y_pred))) / m

    def diff(self, y_pred, y_true):
        return y_pred - y_true


class CategoricalCrossentropy(Loss):
    """Computes the crossentropy loss between the labels and predictions.

      Use this crossentropy loss function when there are two or more label classes.
      We expect labels to be provided in a `one_hot` representation. If you want to
      provide labels as integers, please use `SparseCategoricalCrossentropy` loss.
      There should be `# classes` floating point values per feature.
      Standalone usage:
      ```python
      >>> y_true = [[0, 1, 0], [0, 0, 1]]
      >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
      >>> loss = asf.losses.CategoricalCrossentropy()
      >>> loss(y_pred, y_true).
      1.177
      ```
    """
    __name__ = 'CategoricalCrossentropy'

    def __call__(self, y_pred, y_true):
        return -np.mean(np.log(np.max(y_true * y_pred, axis=1) + 1e-6))

    def diff(self, y_pred, y_true):
        return y_pred - y_true


class SparseCategoricalCrossentropy(Loss):
    """Computes the crossentropy loss between the labels and predictions.

      Use this crossentropy loss function when there are two or more label classes.
      We expect labels to be provided as integers. If you want to provide labels
      using `one-hot` representation, please use `CategoricalCrossentropy` loss.
      There should be `# classes` floating point values per feature for `y_pred`
      and a single floating point value per feature for `y_true`

      Standalone usage:
      ```python
      >>>  y_true = [[1], [2]]
      >>>  y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
      >>>  loss = asf.losses.SparseCategoricalCrossentropy)
      >>>  loss(y_pred,y_true)
      1.177
      ```
    """
    __name__ = 'SparseCategoricalCrossentropy'

    def __call__(self, y_pred, y_true):
        n_c = y_pred.shape[-1]
        y_true = true_one_hot(y_true, n_c)
        return -np.mean(np.log(np.max(y_true * y_pred, axis=1)))

    def diff(self, y_pred, y_true):
        n_c = y_pred.shape[-1]
        y_true = true_one_hot(y_true, n_c)
        return y_pred - y_true
