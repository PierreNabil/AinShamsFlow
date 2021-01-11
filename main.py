"""Used to test all created objects."""

import ainshamsflow as asf
import numpy as np


def _true_one_hot(y_true):
	n_c = np.max(y_true) + 1
	return np.squeeze(np.eye(n_c)[y_true])


x = np.random.rand(15, 10, 10, 3)
y = np.random.randint(0, 5, (15, 1))
y = _true_one_hot(y)
# print(x[0,:,:,0], y[0])
print(x.shape, y.shape)


model = asf.models.Sequential([
	asf.layers.Conv2D(5, 3, padding='same', activation='relu'),
	asf.layers.Pool2D(2),
	asf.layers.Flatten(),
	asf.layers.Dense(100, activation='relu'),
	asf.layers.Dense( 30, activation='relu'),
	asf.layers.Dense( 5, activation='softmax')
], input_shape=(10, 10, 3), name='my_model')

model.print_summary()

model.compile(
	asf.optimizers.SGD(lr=0.01),
	asf.losses.CategoricalCrossentropy()
)

history = model.fit(x, y, 100)
model.evaluate(x, y)
model.evaluate(x, y)

history.show()


# x_test = np.array([[1, 1, 1]])
# y_test = np.array([[10]])
#
# print(model.predict(x_test))
#
# model.evaluate(x_test, y_test)
