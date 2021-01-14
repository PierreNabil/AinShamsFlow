"""Used to test all created objects."""

import ainshamsflow as asf
import numpy as np


x = np.random.rand(15, 10, 10, 3)
y = np.random.randint(0, 5, (15, 1))

ds = asf.data.Dataset(x,y)

conv_part = asf.models.Sequential([
	asf.layers.Conv2D(5, 3, padding='same', activation='relu'),
	asf.layers.Pool2D(2),
	asf.layers.Flatten()
], input_shape=(10, 10, 3), name='conv_part')

conv_part.print_summary()


model = asf.models.Sequential([
	conv_part,
	asf.layers.Dense(100, activation='relu'),
	asf.layers.Dense( 30, activation='relu'),
	asf.layers.Dense(  5, activation='softmax')
], input_shape=(10, 10, 3), name='my_model')

model.print_summary()

model.compile(
	asf.optimizers.AdaGrad(lr=0.1),
	'SparseCategoricalCrossentropy',
	['accuracy', 'precision', 'recall', 'f1score'],
	'l1'
)

history = model.fit(
	ds,
	epochs=50,
	batch_size=4
)
model.evaluate(ds)

history.show()
