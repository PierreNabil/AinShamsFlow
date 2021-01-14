"""History Object Moeule.

In this Module, we define the History Class used to visualize statistics
such as loss and metric values during training.
"""


import numpy as np
import matplotlib.pyplot as plt


class History:
	"""History Class."""

	def __init__(self, loss, metrics=None):
		"""Initialize empty loss and metrics values and save loss and metrics names."""

		if metrics is None:
			metrics = []
		self.loss_values = []
		self.metric_values = []
		self.loss_name = loss.__name__
		self.metric_names = [metric.__name__ for metric in metrics]

		if len(self.loss_name) > 12:
			if self.loss_name[-12:] == 'Crossentropy':
				self.loss_name = 'Crossentropy'

	def add(self, loss_value, metric_values):
		"""Add Values to the history of the training session."""

		self.loss_values.append(loss_value)
		self.metric_values.append(metric_values)

	def flipped_metrics(self):
		"""Return the metric Values as flipped for easy visulaization and data extraction."""

		return np.array(self.metric_values).T

	def show(self, show_metrics=True):
		"""Show the training statistics in an image from saved data."""

		num_of_plots = len(self.metric_names) + 1 if show_metrics else 1

		fig, axs = plt.subplots(num_of_plots, 1, squeeze=False, figsize=(8, 3*num_of_plots))

		# Loss Plots:
		epochs = np.arange(len(self.loss_values)) + 1
		axs[0, 0].plot(epochs, self.loss_values)
		axs[0, 0].set_title('Model Loss')
		axs[0, 0].set_ylabel(self.loss_name)
		axs[0, 0].set_xlabel('epochs')
		bottom, top = axs[0, 0].get_ylim()
		axs[0, 0].set_ylim(0, top)

		# Accuracy Plots:
		if self.metric_names and show_metrics:
			for j, metric_values in enumerate(self.flipped_metrics()):
				axs[j+1, 0].plot(metric_values)
				if j == 0:
					axs[j+1, 0].set_title('Model Metrics')
				axs[j+1, 0].set_ylabel(self.metric_names[j])
				axs[j+1, 0].set_xlabel('epochs')
				axs[j+1, 0].set_ylim(0, 1)
		plt.show()
