import numpy as np
import matplotlib.pyplot as plt


class History:
	def __init__(self, loss, metrics=[]):
		self.loss_values = []	#(num_of_terations,)
		self.metric_values = []	#(,)
		self.loss_name = loss.name
		self.metric_names = [metric.name for metric in metrics]

	def add(self, loss_value, metric_values):
		self.loss_values.append(loss_value)
		self.metric_values.append(metric_values)

	def _fliped_metrics(self):
		return np.array(self.metric_values).T

	def show(self, show_metrics=True):
		num_of_plots = len(self.metric_names) + 1 if show_metrics else 1

		fig, axs = plt.subplots(num_of_plots, 1, squeeze=False)

		# Loss Plots:
		axs[0, 0].plot(self.loss_values)
		axs[0, 0].set_title('Model Loss')
		axs[0, 0].set_ylabel(self.loss_name)
		axs[0, 0].set_xlabel('epochs')
		bottom, top = axs[0, 0].get_ylim()
		axs[0, 0].set_ylim(0, top)

		# Accuracy Plots:
		if self.metric_names and show_metrics:
			for j, metric_values in enumerate(self._fliped_metrics()):
				axs[j+1, 0].plot(metric_values)
				if j==0:
					axs[j+1, 0].set_title('Model Metrics')
				axs[j+1, 0].set_ylabel(self.metric_names[j])
				axs[j+1, 0].set_xlabel('epochs')
				axs[j+1, 0].set_ylim(0, 1)
		plt.show()
