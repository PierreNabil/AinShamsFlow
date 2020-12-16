import ainshamsflow as asf
import numpy as np
#TODO: Test what has been done thus far

x = np.arange(10).reshape((1, -1))
y = 2*x+1

print(x)
print(y)

model = asf.models.Sequential([
	asf.layers.Dense(3, asf.activations.ReLU(), 'layer_1'),
	asf.layers.Dense(1, asf.activations.Linear(), 'layer_2')
], 1, 'simple_model')

model.compile(
	optimizer=asf.optimizers.SGD(lr=0.02),
	loss=asf.losses.MSE(),
	metrics=[asf.metrics.SoftAccuracy(), asf.metrics.HardAccuracy()],
	regularizer=asf.regularizers.L2(1)
)

model.summary()

history = model.fit(x, y, 100, 3)
history.show()
model.predict(x)
model.evaluate(x, y, 3)
