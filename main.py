"""Used to test all created objects."""

import ainshamsflow as asf
import numpy as np

x = np.arange(10).reshape((-1, 1))
y = 2*x+1

print(y)

model_layer = asf.models.Sequential([
	asf.layers.Dense(5),
	asf.layers.Dense(3),
	asf.layers.Activation('leakyrelu')
], input_shape=(8,), name='model_layer')

model = asf.models.Sequential([
	asf.layers.Dense(8),
	asf.layers.Reshape((2, 2, -1)),
	asf.layers.BatchNorm(),
	asf.layers.Dropout(0.9),
	asf.layers.Flatten(),
	model_layer,
	asf.layers.Dense(1)
], input_shape=(1,), name='my_model')

model.compile(
	optimizer=asf.optimizers.SGD(lr=0.0000001),
	loss=asf.losses.MSE(),
	metrics=[asf.losses.MAE()]
)

model.print_summary()

history = model.fit(x, y, 100, verbose=False)
history.show()
print(model.predict(x))
model.evaluate(x, y, 3)

model.save_model('.\\test_model')

model2 = asf.models.load_model('.\\test_model\\my_model')

model2.print_summary()