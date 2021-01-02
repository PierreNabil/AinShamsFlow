"""Used to test all created objects."""

import ainshamsflow as asf
import numpy as np

x = np.random.randint(-5, 5, (10, 3))
y = np.sum(np.square(x) * np.array([[2, 3, 5]]), axis=1, keepdims=True)
print(x[0], y[0])
print(x.shape, y.shape)


model = asf.models.Sequential([
	asf.layers.Dense(5, name='fc_1'),
	asf.layers.BatchNorm(name='bn_1'),
	asf.layers.Activation('relu'),
	asf.layers.Dense(3, name='fc_2'),
	asf.layers.BatchNorm(name='bn_2'),
	asf.layers.Activation('relu'),
	asf.layers.Dense(1, name='fc_3'),
	asf.layers.BatchNorm(name='bn_3'),
	asf.layers.Activation('linear')
], input_shape=(3,), name='my_model')

model.print_summary()

model.compile(
	asf.optimizers.RMSProp(lr=0.001),
	asf.losses.MSE()
)

history = model.fit(x, y, 10)
model.evaluate(x, y)

history.show()


x_test = np.array([[1, 1, 1]])
y_test = np.array([[10]])

print(model.predict(x_test))

model.evaluate(x_test, y_test)
