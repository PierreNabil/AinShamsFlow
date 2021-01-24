"""Used to test all created objects."""

import ainshamsflow as asf
from sklearn.datasets import fetch_openml

ds_split = 40000

# Get the Dataset
openml_ds = fetch_openml('mnist_784', version=1)
X, Y = openml_ds["data"], openml_ds["target"]

X_train = X[:ds_split] / 255
X_valid = X[ds_split:] / 255
y_train = Y[:ds_split].astype('int').reshape(-1, 1)
y_valid = Y[ds_split:].astype('int').reshape(-1, 1)

ds_train = asf.data.Dataset(X_train, y_train)
ds_valid = asf.data.Dataset(X_valid, y_valid)

# Create the Model
model = asf.models.Sequential([
	asf.layers.Dense(300, activation='relu'),
	asf.layers.Dense(100, activation='relu'),
	asf.layers.Dense( 10, activation='softmax')
], input_shape=(784,), name='DNN_model')

model.print_summary()

# Train the Model
model.compile(
	asf.optimizers.AdaGrad(lr=0.01),
	'sparsecategoricalcrossentropy',
	['accuracy']
)

history = model.fit(
	ds_train,
	epochs=20,
	valid_data=ds_valid,
	batch_size=1024,
	valid_batch_size=512
)

# Model Evaluation
history.show()

model.evaluate(ds_valid)
