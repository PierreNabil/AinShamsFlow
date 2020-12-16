import matplotlib.pyplot as plt


class History:
	def __init__(self, loss, metrics=[]):
		self.loss_values = []
		self.metric_values = []
		self.loss_name = loss.name
		self.metric_names = [metric.name for metric in metrics]

	def add(self, loss_value, metric_values):
		self.loss_values.extend(loss_value)
		self.metric_values.extend(metric_values)

	def show(self, show_metrics=True):
		#TODO: implement the show function.
		plt.show()
