"""Optimizers Module.

In this Module, we include our optimization algorithms such as
Stocastic Gradient Descent (SGD).
"""

import numpy as np

from ainshamsflow.metrics import Metric
from ainshamsflow.utils.asf_errors import BaseClassError, NameNotFoundError, UnsupportedShapeError
from ainshamsflow.utils.history import History


def get(opt_name):
    """Get any Optimizer in this Module by name."""

    opts = [SGD, Momentum, AdaDelta, RMSProp, Adam, AdaDelta]
    for opt in opts:
        if opt.__name__.lower() == opt_name.lower():
            return opt()
    else:
        raise NameNotFoundError(opt_name, __name__)


def _check_dims(weights, dw):
    if weights.shape != dw.shape:
        raise UnsupportedShapeError(dw.shape, weights.shape)


class Optimizer:
    """Optimiser Base Class.

    Used to generalize learning using Gradient Descent.

    Creating a new optimizer is as easy as creating a new class that
    inherits from this class.
    You then have to add any extra parameters in your constructor
    (while still calling this class' constructor) and redefine the
    _update() method.
    """

    def __init__(self, lr=0.01):
        """Initialize the Learning Rate."""
        self.lr = lr

    def __call__(self, ds, epochs, batch_size, layers, loss, metrics, regularizer, training=True, verbose=True):
        """Run optimization using Gradient Descent. Return History Object for training session."""

        m = ds.cardinality()

        if batch_size is not None and 0 < batch_size < m:
            ds.batch(batch_size)

        history = History(loss, metrics)
        loss_values = []
        metrics_values = []

        if verbose:
            if training:
                print('Training Model for {} epochs:'.format(epochs))
            else:
                print('Evaluating Model:')

        for i, (batch_x, batch_y) in enumerate(ds):
            loss_value, metric_values = self._single_iteration(batch_x, batch_y, m, layers,
                                                               loss, metrics, regularizer, False, i)
            loss_values.append(loss_value)
            metrics_values.append(metric_values)

        loss_values = np.array(loss_values).sum()
        metrics_values = np.array(metrics_values).mean(axis=0)

        if verbose:
            if training:
                print(
                    'Epoch #{:4d}:'.format(0),
                    '{}={:8.4f},'.format(loss.__name__, loss_values),
                    *['{}={:8.4f},'.format(metric.__name__, metrics_values[j]) for j, metric in enumerate(metrics)]
                )
                history.add(loss_values, metrics_values)
            else:
                print(
                    '{}={:8.4f},'.format(loss.__name__, loss_values),
                    *['{}={:8.4f},'.format(metric.__name__, metrics_values[j]) for j, metric in enumerate(metrics)]
                )

        if not training:
            if verbose:
                return
            else:
                return loss_values, metrics_values

        for epoch_num in range(epochs):
            loss_values = []
            metrics_values = []
            for i, (batch_x, batch_y) in enumerate(ds):
                loss_value, metric_values = self._single_iteration(batch_x, batch_y, m, layers,
                                                                   loss, metrics, regularizer, training, i)
                loss_values.append(loss_value)
                metrics_values.append(metric_values)

            loss_values = np.array(loss_values).sum()
            metrics_values = np.array(metrics_values).mean(axis=0)
            history.add(loss_values, metrics_values)
            if verbose:
                print(
                    'Epoch #{:4d}:'.format(epoch_num+1),
                    '{}={:8.4f},'.format(loss.__name__, loss_values),
                    *['{}={:8.4f},'.format(metric.__name__, metrics_values[j]) for j, metric in enumerate(metrics)]
                )

        return history

    def _single_iteration(self, batch_x, batch_y, m, layers, loss, metrics, regularizer, training, i):
        """Run optimization for a single batch. Return loss value and metric values for iteration."""

        weights_list = [layer.get_weights()[0] for layer in layers]
        # Forward Pass
        batch_a = batch_x
        for j, layer in enumerate(layers):
            batch_a = layer(batch_a, training)
        regularization_term = 0 if regularizer is None else regularizer(weights_list, m)
        loss_value = loss(batch_a, batch_y) + regularization_term
        metric_values = []
        if metrics:
            Metric.calc_confusion_matrix(batch_a, batch_y)
            for metric in metrics:
                metric_values.append(metric())
        # Backward Pass
        regularization_diff = None if regularizer is None else regularizer.diff(weights_list, m)
        da = loss.diff(batch_a, batch_y)
        if training:
            for j in reversed(range(len(layers))):
                da, dw, db = layers[j].diff(da)
                if layers[j].trainable:
                    if regularization_diff is not None:
                        dw = self._add_reg_diff(dw, regularization_diff[j])
                    weights, biases = layers[j].get_weights()
                    updated_weights = self._update(i, weights, dw, layer_num=j, is_weight=True)
                    updated_biases = self._update(i, biases, db, layer_num=j, is_weight=False)
                    layers[j].set_weights(updated_weights, updated_biases)
        return loss_value, metric_values

    def _update(self, i, weights, dw, layer_num, is_weight):
        raise BaseClassError

    def _add_reg_diff(self, dw, reg_diff):
        if isinstance(dw, list) and isinstance(reg_diff, list):
            dw_new = []
            for single_dw, single_reg_diff in zip(dw, reg_diff):
                single_dw = self._add_reg_diff(single_dw, single_reg_diff)
                dw_new.append(single_dw)
        else:
            dw_new = dw + reg_diff
        return dw_new


class SGD(Optimizer):
    """Stochastic Gradient Descent Algorithm."""

    __name__ = 'SGD'

    def _update(self, i, weights, dw, layer_num, is_weight):
        """Update step for SGD."""
        if isinstance(weights, list) and isinstance(dw, list):
            ans = []
            for j, (weight, d) in enumerate(zip(weights, dw)):
                l_num = '{}.{}'.format(layer_num, j)
                ans.append(self._update(i, weight, d, l_num, is_weight))
            return ans
        else:
            _check_dims(weights, dw)

            return weights - self.lr * dw


class Momentum(Optimizer):
    """SGD with Momentum"""

    __name__ = 'Momentum'

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.momentum = {}

    def _update(self, i, weights, dw, layer_num, is_weight):
        if isinstance(weights, list) and isinstance(dw, list):
            ans = []
            for j, (weight, d) in enumerate(zip(weights, dw)):
                l_num = '{}.{}'.format(layer_num, j)
                ans.append(self._update(i, weight, d, l_num, is_weight))
            return ans
        else:
            _check_dims(weights, dw)

            # check for first time:
            layer_id = "{}{}".format(layer_num, 'w' if is_weight else 'b')
            if layer_id not in self.momentum.keys():
                self.momentum[layer_id] = np.zeros(weights.shape)

            # update step
            self.momentum[layer_id] = self.beta * self.momentum[layer_id] + (1 - self.beta) * dw

            return weights - self.lr * self.momentum[layer_id]


class AdaGrad(Optimizer):
    """AdaGrad"""

    __name__ = 'AdaGrad'

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.RMS = {}

    def _update(self, i, weights, dw, layer_num, is_weight):
        if isinstance(weights, list) and isinstance(dw, list):
            ans = []
            for j, (weight, d) in enumerate(zip(weights, dw)):
                l_num = '{}.{}'.format(layer_num, j)
                ans.append(self._update(i, weight, d, l_num, is_weight))
            return ans
        else:
            _check_dims(weights, dw)

            # check for first time:
            layer_id = "{}{}".format(layer_num, 'w' if is_weight else 'b')
            if layer_id not in self.RMS.keys():
                self.RMS[layer_id] = np.zeros(weights.shape)

            # update step
            self.RMS[layer_id] = self.RMS[layer_id] + np.square(dw)

            return weights - self.lr * dw / (np.sqrt(self.RMS[layer_id]) + 1e-8)


class AdaDelta(Optimizer):
    """AdaDelta"""

    __name__ = 'AdaDelta'

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.delta = {}

    def _update(self, i, weights, dw, layer_num, is_weight):
        if isinstance(weights, list) and isinstance(dw, list):
            ans = []
            for j, (weight, d) in enumerate(zip(weights, dw)):
                l_num = '{}.{}'.format(layer_num, j)
                ans.append(self._update(i, weight, d, l_num, is_weight))
            return ans
        else:
            _check_dims(weights, dw)

            # check for first time:
            layer_id = "{}{}".format(layer_num, 'w' if is_weight else 'b')
            if layer_id not in self.delta.keys():
                self.delta[layer_id] = np.zeros(weights.shape)

            # update step
            self.delta[layer_id] = self.beta * self.delta[layer_id] + (1 - self.beta) * np.square(dw)

            return weights - self.lr * dw / (np.sqrt(self.delta[layer_id]) + 1e-8)


class RMSProp(Optimizer):
    """RMSProp"""

    __name__ = 'RMSProp'

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.stMoment = {}
        self.ndMoment = {}
        self.RMS = {}

    def _update(self, i, weights, dw, layer_num, is_weight):
        if isinstance(weights, list) and isinstance(dw, list):
            ans = []
            for j, (weight, d) in enumerate(zip(weights, dw)):
                l_num = '{}.{}'.format(layer_num, j)
                ans.append(self._update(i, weight, d, l_num, is_weight))
            return ans
        else:
            _check_dims(weights, dw)

            # check for first time:
            layer_id = "{}{}".format(layer_num, 'w' if is_weight else 'b')
            if layer_id not in self.stMoment.keys():
                self.stMoment[layer_id] = np.zeros(weights.shape)
            if layer_id not in self.ndMoment.keys():
                self.ndMoment[layer_id] = np.zeros(weights.shape)
            if layer_id not in self.RMS.keys():
                self.RMS[layer_id] = np.zeros(weights.shape)

            # update step
            self.stMoment[layer_id] = self.beta * self.stMoment[layer_id] + (1 - self.beta) * dw
            self.ndMoment[layer_id] = self.beta * self.ndMoment[layer_id] + (1 - self.beta) * np.square(dw)
            self.RMS[layer_id] = self.beta * self.RMS[layer_id] + self.lr * dw / (
                np.sqrt(self.ndMoment[layer_id] - np.square(self.stMoment[layer_id]) + 1e-8))

            return weights - self.RMS[layer_id]


class Adam(Optimizer):
    """Adam Optimizer"""

    __name__ = 'Adam'

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = {}
        self.secMoment = {}

    def _update(self, i, weights, dw, layer_num, is_weight):
        if isinstance(weights, list) and isinstance(dw, list):
            ans = []
            for j, (weight, d) in enumerate(zip(weights, dw)):
                l_num = '{}.{}'.format(layer_num, j)
                ans.append(self._update(i, weight, d, l_num, is_weight))
            return ans
        else:
            _check_dims(weights, dw)

            # check for first time:
            layer_id = "{}{}".format(layer_num, 'w' if is_weight else 'b')
            if layer_id not in self.secMoment.keys():
                self.secMoment[layer_id] = np.zeros(weights.shape)
            if layer_id not in self.momentum.keys():
                self.momentum[layer_id] = np.zeros(weights.shape)

            # update step
            self.secMoment[layer_id] = (self.beta2 * self.secMoment[layer_id] + (1 - self.beta2) * np.square(dw)) / (
                    1 - np.power(self.beta2, i + 1))
            self.momentum[layer_id] = (self.beta1 * self.momentum[layer_id] + (1 - self.beta1) * dw) / (
                    1 - np.power(self.beta1, i + 1))

            return weights - self.lr * self.momentum[layer_id] / (np.sqrt(self.secMoment[layer_id]) + 1e-8)
