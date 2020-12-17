import ainshamsflow as asf
import numpy as np
#TODO: Test what has been done thus far

x = np.arange(10).reshape((-1, 1))
y = 2*x+1

model = asf.models.Sequential([
	asf.layers.Dense(8, name='layer_1'),
	asf.layers.Reshape((2, 2, 2), name='reshape_1'),
	asf.layers.Dropout(0.9, name='dropout_1'),
	asf.layers.Reshape((8,), name='reshape_2'),
	asf.layers.Dense(1, name='layer_2')
], input_shape=(1,), name='simple_model')

model.compile(
	optimizer=asf.optimizers.SGD(lr=0.00001),
	loss=asf.losses.MSE(),
	metrics=[asf.losses.MAPE()]
)

model.print_summary()

history = model.fit(x, y, 100, verbose=False)
history.show()
model.predict(x)
model.evaluate(x, y, 3)
