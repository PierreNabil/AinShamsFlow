"""Utils module for miscellaneous helper functions."""

import numpy as np

from ainshamsflow.data import Dataset


def pred_one_hot(y_pred):
	n_c = y_pred.shape[-1]
	return np.squeeze(np.eye(n_c)[np.argmax(y_pred, axis=-1)])


def true_one_hot(y_true, n_c):
	m = y_true.shape[0]
	return np.squeeze(np.eye(n_c)[y_true]).reshape((m, -1))


def confution_matrix(y_pred_1h, y_true_1h):
	y_pred = y_pred_1h.argmax(axis=1)
	y_true = y_true_1h.argmax(axis=1)

	m, n_c = y_pred_1h.shape

	cm = np.zeros((n_c, n_c))

	for i in range(m):
		cm[y_true[i]][y_pred[i]] += 1

	return cm


def get_dataset_from_xy(x, y):
	if x is None:
		raise RunningWithoutDataError

	elif isinstance(x, Dataset):
		if x.data is None:
			raise RunningWithoutDataError

		elif x.target is None:
			if y is None:
				raise RunningWithoutDataError
			elif isinstance(y, Dataset):
				return x.add_targets(y.target)
			else: # isinstance(y, np.array)
				return x.add_targets(y)

		else:  # x.target is not None
			return x

	else:  # isinstance(x, np.array)
		if y is None:
			raise RunningWithoutDataError
		elif isinstance(y, Dataset):
			return y.add_data(x)
		else: # isinstance(y, np.array)
			return Dataset(x, y)


def get_indices(X_shape, HF, WF, stride, pad):
	"""
        Returns index matrices in order to transform our input image into a matrix.
        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.
        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d.
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
	# get input size
	m, n_H, n_W, n_C = X_shape
	p_h, p_w = pad
	s_h, s_w = stride

	# get output size
	out_h = int((n_H + 2 * p_h - HF) / s_h) + 1
	out_w = int((n_W + 2 * p_w - WF) / s_w) + 1

	# ----Compute matrix of index i----

	# Level 1 vector.
	level1 = np.repeat(np.arange(HF), WF)
	# Duplicate for the other channels.
	level1 = np.tile(level1, n_C)
	# Create a vector with an increase by 1 at each level.
	everyLevels = s_h * np.repeat(np.arange(out_h), out_w)
	# Create matrix of index i at every levels for each channel.
	i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

	# ----Compute matrix of index j----

	# Slide 1 vector.
	slide1 = np.tile(np.arange(WF), HF)
	# Duplicate for the other channels.
	slide1 = np.tile(slide1, n_C)
	# Create a vector with an increase by 1 at each slide.
	everySlides = s_h * np.tile(np.arange(out_w), out_h)
	# Create matrix of index j at every slides for each channel.
	j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

	# ----Compute matrix of index d----

	# This is to mark delimitation for each channel
	# during multi-dimensional arrays indexing.
	d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

	return i, j, d


def im2col(X, idx, pad):
	"""
        Transforms our input image into a matrix.
        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.
        Returns:
        -cols: output matrix.
    """
	p_h, p_w = pad
	# Padding
	X_padded = np.pad(X, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode='constant')
	i, j, d = idx
	# Multi-dimensional arrays indexing.
	cols = X_padded[:, d, i, j]
	cols = np.concatenate(cols, axis=-1)
	return cols


def col2im(dX_col, X_shape,idx,  pad):
	"""
        Transform our matrix back to the input image.
        Parameters:
        - dX_col: matrix with error.
        - X_shape: input image shape.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.
        Returns:
        -x_padded: input image with error.
    """
	# Get input size
	N, D, H, W = X_shape
	p_h, p_w = pad
	# Add padding if needed.
	H_padded, W_padded = H + 2 * p_h, W + 2 * p_w
	X_padded = np.zeros((N, D, H_padded, W_padded))

	# Index matrices, necessary to transform our input image into a matrix.
	i, j, d = idx
	# Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
	dX_col_reshaped = np.array(np.hsplit(dX_col, N))
	# Reshape our matrix back to image.
	# slice(None) is used to produce the [::] effect which means "for every elements".
	np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
	# Remove padding from new image if needed.
	if all((p_h == 0, p_w == 0)):
		return X_padded
	else:
		return X_padded[p_h:-p_h, p_w:-p_w, :, :]


def time_elapsed(t):
	if t > 60:
		return '{:.3f} mins'.format(t/60)
	elif t > 0:
		return '{:.3f} s'.format(t)
	elif t > 1e-3:
		return '{:.3f} ms'.format(t*1e3)
	else:
		return '{} us'.format(t*1e6)
