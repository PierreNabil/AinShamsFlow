"""AinShamsFlow (asf) Deep Learning Framework.

# AinShamsFlow
4th CSE Neural Networks Project.

## Contents:
* [Project Description](#Project-Description)
* [Install](#Install)
* [Usage](#Usage)
* [Team Members](#Team-Members)

## Project Description:
AinShamsFlow (asf) is a Deep Learning Framework built by our [Team](#Team-Members)
from Ain Shams University during December 2020 - January 2021.

The Design and Interface is inspired heavily from Keras and TensorFlow.

However, we only implement everything from scratch using only simple libraries
such as numpy and matplotlib.

## Install:
You can install the latest available version as follows:
```shell
$ pip install ainshamsflow
```

## Usage:
you start using this project my importing the package as follows:

```python
>>> import ainshamsflow as asf
```

then you can start creating a model:

```python
>>> model = asf.models.Sequential([
...     asf.layers.Dense(300, activation="relu"),
...     asf.layers.Dense(100, activation="relu"),
...     asf.layers.Dense( 10, activation="softmax")
... ], input_shape=(784,), name="my_model")
>>> model.print_summary()
```

then compile and train the model:

```python
>>> model.compile(
... 	optimizer=asf.optimizers.SGD(lr=0.01),
... 	loss='sparsecategoricalcrossentropy',
...     metrics=['accuracy']
... )
>>> history = model.fit(X_train, y_train, epochs=100)
```

finally you can evaluate, predict and show training statistics:

```python
>>> history.show()
>>> model.evaluate(X_valid, y_valid)
>>> y_pred = model.predict(X_test)
```

A more elaborate example usage can be found in this
[demo notebook](https://colab.research.google.com/drive/1sqEeUUkG3bTplhlLb73QCGUbaubX2aXi?usp=sharing).

## Team Members:
* [Pierre Nabil](https://github.com/PierreNabil)
* [Ahmed Taha](https://github.com/atf01)
* [Girgis Micheal](https://github.com/girgismicheal)
* [Hazzem Mohammed](https://github.com/hazzum)
* [Ibrahim Shoukry](https://github.com/IbrahimShoukry512)
* [John Bahaa](https://github.com/John-Bahaa)
* [Michael Magdy](https://github.com/Michael-M-Mike)
* [Ziad Tarek](https://github.com/ziadtarekk)


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
import ainshamsflow.optimizers as optimizers
import ainshamsflow.regularizers as regularizers

__all__ = [
	'activations',
	'data',
	'initializers',
	'layers',
	'losses',
	'metrics',
	'models',
	'optimizers',
	'regularizers'
]
