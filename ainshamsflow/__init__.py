"""AinShamsFlow (asf) Deep Learning Framework.

AinShamsFlow is a simple Deep Learning Framework inspired by the Keras API.
asf was created by a team of 8 undergraduate students from Ain Shams University
in Cairo, Egypt in December 2020 as a project for the Neural Networks subject
taken during the senior year.

asf is implemented using only simple libraries such as numpy and matplotlib.


you start using this project my importing the package as follows:

import ainshamsflow as asf

then you can start creating a model:

model = asf.models.Sequential([
	asf.layers.Dense(30, activation="relu"),
	asf.layers.Dense(10, activation="relu"),
	asf.layers.Dense( 1, activation="relu")
], input_shape=(100,), name='my_model')
model.print_summary()

then compile and train the model:

model.compile(
	optimizer=asf.optimizers.SGD(),
	loss=asf.losses.MSE()
)
history = model.fit(X, y, epochs=100)

finally you can evaluate, predict and show training statistics:

model.evaluate(X, y)
y_pred = model.predict(X_val)
history.show()


You can find find a better usage example in the README and main.py if the
GitHub Repository for this project here:
https://github.com/PierreNabil/AinShamsFlow
"""

import ainshamsflow.activations as activations
import ainshamsflow.data as data
import ainshamsflow.initializers as initializers
import ainshamsflow.layers as layers
import ainshamsflow.losses as losses
import ainshamsflow.metrics as metrics
import ainshamsflow.models as models
import ainshamsflow.optimizers as optimizsers
import ainshamsflow.regularizers as regularizers