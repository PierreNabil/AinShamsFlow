"""History Object Moeule.

In this Module, we define the History Class used to visualize statistics
such as loss and metric values during training.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class History:
	"""History Class."""

	def __init__(self, loss, metrics=None, valid=False):
		"""Initialize empty loss and metrics values and save loss and metrics names."""

		if metrics is None:
			metrics = []

		self.valid = valid

		self.loss_values = []
		self.metric_values = []
		if valid:
			self.val_loss_values = []
			self.val_metric_values = []

		self.loss_name = loss.__name__
		self.metric_names = [metric.__name__ for metric in metrics]

		if len(self.loss_name) > 12:
			if self.loss_name[-12:] == 'Crossentropy':
				self.loss_name = 'Crossentropy'

	def add(self, loss_value, metric_values, val_loss_value=None, val_metric_values=None):
		"""Add Values to the history of the training session."""

		self.loss_values.append(loss_value)
		self.metric_values.append(metric_values)

		if self.valid:
			self.val_loss_values.append(val_loss_value)
			self.val_metric_values.append(val_metric_values)


	def flipped_metrics(self):
		"""Return the metric Values as flipped for easy visualization and data extraction."""

		return np.array(self.metric_values).T

	def flipped_valid_metrics(self):
		"""Return the metric Values as flipped for easy visualization and data extraction."""

		assert self.valid
		return np.concatenate([
			np.array(self.metric_values).T,
			np.array(self.val_metric_values).T
		], axis=0)


	def show(self, show_metrics=True):
		"""Show the training statistics in an image from saved data."""

		if self.valid:
			num_of_plots = len(self.metric_names) + 1 if show_metrics else 1
			num_of_plots *= 2

			fig, axs = plt.subplots(num_of_plots, 1, squeeze=False, figsize=(8, 3 * num_of_plots))
			fig.patch.set_facecolor('w')

			# Loss Plots:

			axs[0, 0].plot(self.loss_values)
			axs[0, 0].set_title('Model Loss')
			axs[0, 0].set_ylabel('loss')
			bottom, top = axs[0, 0].get_ylim()
			axs[0, 0].set_ylim(0, top)
			axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

			axs[1, 0].plot(self.val_loss_values)
			axs[1, 0].set_ylabel('val_loss')
			axs[1, 0].set_xlabel('epochs')
			bottom, top = axs[1, 0].get_ylim()
			axs[1, 0].set_ylim(0, top)
			axs[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

			# Metrics Plots:
			if self.metric_names and show_metrics:
				for j, (metric_name, metric_values, valid_values) in enumerate(zip(self.metric_names, self.flipped_metrics(), self.flipped_valid_metrics())):
					if j == 0:
						axs[2*j + 2, 0].set_title('Model Metrics')

					axs[2*j + 2, 0].plot(metric_values)
					axs[2*j + 3, 0].plot(valid_values)

					axs[2*j + 2, 0].set_ylabel(metric_name)
					axs[2*j + 3, 0].set_ylabel('val_' + metric_name)

					if len(metric_name) == 2:
						bottom, top = axs[2*j + 2, 0].get_ylim()
						axs[2*j + 2, 0].set_ylim(0, top)
						bottom, top = axs[2*j + 3, 0].get_ylim()
						axs[2*j + 3, 0].set_ylim(0, top)
					else:
						axs[2*j + 2, 0].set_ylim(0, 1)
						axs[2*j + 3, 0].set_ylim(0, 1)

					axs[2*j + 2, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
					axs[2*j + 3, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

				axs[-1, 0].set_xlabel('epochs')


		else:
			num_of_plots = len(self.metric_names) + 1 if show_metrics else 1

			fig, axs = plt.subplots(num_of_plots, 1, squeeze=False, figsize=(8, 3*num_of_plots))
			fig.patch.set_facecolor('w')

			# Loss Plots:
			axs[0, 0].plot(self.loss_values)
			axs[0, 0].set_title('Model Loss')
			axs[0, 0].set_ylabel('loss')
			axs[0, 0].set_xlabel('epochs')
			bottom, top = axs[0, 0].get_ylim()
			axs[0, 0].set_ylim(0, top)
			axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

			# Metrics Plots:
			if self.metric_names and show_metrics:
				for j, metric_values in enumerate(self.flipped_metrics()):
					axs[j+1, 0].plot(metric_values)
					if j == 0:
						axs[j+1, 0].set_title('Model Metrics')
					axs[j+1, 0].set_ylabel(self.metric_names[j])
					if len(self.metric_names[j]) == 2:
						bottom, top = axs[j+1, 0].get_ylim()
						axs[j+1, 0].set_ylim(0, top)
					else:
						axs[j+1, 0].set_ylim(0, 1)
					axs[j+1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
				axs[-1, 0].set_xlabel('epochs')


		plt.show()
