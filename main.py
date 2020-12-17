import ainshamsflow as asf
import numpy as np
#TODO: Test what has been done thus far

x = np.arange(10).reshape((-1, 1))
y = 2*x+1

print(y)

model = asf.models.Sequential([
	asf.layers.Dense(8, name='layer_1'),
	asf.layers.Reshape((2, 2, 2), name='reshape_1'),
	asf.layers.BatchNorm('b_norm_1'),
	asf.layers.Dropout(0.9, name='dropout_1'),
	asf.layers.Flatten(name='flatten'),
	asf.layers.Dense(1, name='layer_2'),
	asf.layers.Activation('leakyrelu')
], input_shape=(1,), name='simple_model')

model.compile(
	optimizer=asf.optimizers.SGD(lr=0.001),
	loss=asf.losses.MSE(),
	metrics=[asf.losses.MAE()]
)

model.print_summary()

history = model.fit(x, y, 100, verbose=False)
history.show()
print(model.predict(x))
model.evaluate(x, y, 3)
