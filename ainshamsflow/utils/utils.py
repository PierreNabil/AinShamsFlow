"""Utils module for miscellaneous helper functions."""

import numpy as np


def pred_one_hot(y_pred):
    n_c = y_pred.shape[-1]
    return np.squeeze(np.eye(n_c)[np.argmax(y_pred, axis=-1)])


def true_one_hot(y_true, n_c):
    return np.squeeze(np.eye(n_c)[y_true])
