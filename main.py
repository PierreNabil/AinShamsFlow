import ainshamsflow as asf
import numpy as np
#TODO: Test what has been done thus far

x = np.arange(10).reshape((-1, 1))
y = 2*x+1

print(y)

model = asf.models.Sequential([
	asf.layers.Dense(8),
	asf.layers.Reshape((2, 2, 2)),
	asf.layers.BatchNorm(),
	asf.layers.Dropout(0.9),
	asf.layers.Flatten(),
	asf.layers.Dense(1),
	asf.layers.Activation('leakyrelu')
], input_shape=(1,))

model.compile(
	optimizer=asf.optimizers.SGD(lr=0.0001),
	loss=asf.losses.MSE(),
	metrics=[asf.losses.MAE()]
)

model.print_summary()

history = model.fit(x, y, 100, verbose=False)
history.show()
print(model.predict(x))
model.evaluate(x, y, 3)
