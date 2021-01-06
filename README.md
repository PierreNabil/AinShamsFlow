# AinShamsFlow
4th CSE Neural Networks Project.

## Contents:
* [Project Description](#Project-Description)
* [Project Structure](#Project-Structure)
* [Usage](#Usage)
* [Team Members](#Team-Members)
* [Todo](#Todo)

## Project Description:
AinShamsFlow (asf) is a Deep Learning Framework built by our [Team](#Team-Members) from Ain Shams University in December 2020.
The Design and Interface is inspired heavily from Keras and TensorFlow.
However, we only implement everything from scratch using only simple libraries such as numpy and matplotlib.

## Project Structure:
The Design of all parts can be seen in the following UML Diagram.

![UML Diagram of the Project](/images/UML%20Class%20Diagram.png)

This is how the Design Structure should work in our Framework.

![Desgin Structure](/images/Design%20Structure.png)

## Usage:
you start using this project my importing the package as follows:

```python
import ainshamsflow as asf
```

then you can start creating a model:

```python
model = asf.models.Sequential([
	asf.layers.Dense(30, activation="relu"),
	asf.layers.Dense(10, activation="relu"),
	asf.layers.Dense( 1, activation="relu")
], input_shape=(100,), name='my_model')
model.print_summary()
```

then compile and train the model:

```python
model.compile(
	optimizer=asf.optimizers.SGD(),
	loss=asf.losses.MSE()
)
history = model.fit(X, y, epochs=100)
```

finally you can evaluate, predict and show training statistics:

```python
model.evaluate(X, y)
y_pred = model.predict(X_val)
history.show()
```


A more elaborate example usage can be found in [main.py](https://github.com/PierreNabil/AinShamsFlow/blob/master/main.py)


## Team Members:
* [Pierre Nabil](https://github.com/PierreNabil)

* [Ahmed Taha](https://github.com/atf01)

* [Girgis Micheal](https://github.com/girgismicheal)

* [Hazzem Mohammed](https://github.com/hazzum)

* [Ibrahim Shoukry](https://github.com/IbrahimShoukry512)

* [John Bahaa](https://github.com/John-Bahaa)

* [Michael Magdy](https://github.com/Michael-M-Mike)

* [Ziad Tarek](https://github.com/ziadtarekk)


## Todo:
### Framework Design:
- [ ] A Data Module to read and process datasets.
    - [ ] Dataset
        - [ ] \_\_init\_\_()
        - [ ] \_\_str\_\_()
        - [ ] \_\_bool\_\_()
        - [ ] \_\_len\_\_()
        - [ ] \_\_iter\_\_()
        - [ ] \_\_next\_\_()
        - [ ] apply()
        - [ ] numpy()
        - [ ] batch()
        - [ ] cardinality()
        - [ ] concatenate()
        - [ ] enumerate()
        - [ ] filter()
        - [ ] from_iterator()
        - [ ] map()
        - [ ] range()
        - [ ] reduce()
        - [ ] shuffle()
        - [ ] skip()
        - [ ] take()
        - [ ] unbatch()
        - [ ] zip()
    - [ ] ImageDataGenerator
        - [ ] \_\_init\_\_()
        - [ ] flow_from_directory()

- [ ] A NN Module to design different architectures.
    - [x] Activation Functions
        - [x] Linear
        - [x] Sigmoid
        - [x] Hard Sigmoid
        - [x] Tanh
        - [x] Hard Tanh
        - [x] ReLU
        - [x] LeakyReLU
        - [x] ELU
        - [x] SELU
        - [x] Softmax
        - [x] Softplus
        - [x] Softsign
        - [x] Swish

    - [x] Layers
    
        DNN Layers:
        - [x] Dense
        - [x] BatchNorm
        - [x] Dropout
        
        CNN Layers 1D: (optional)
        - [x] Conv
        - [x] Pool (Avg and Max)
        - [x] GlobalPool (Avg and Max)
        - [x] Upsample
        
        CNN Layers 2D:
        - [x] Conv
        - [x] Pool (Avg and Max)
        - [x] GlobalPool (Avg and Max)
        - [x] Upsample
        
        CNN Layers 3D: (optional)
        - [x] Conv
        - [x] Pool (Avg and Max)
        - [x] GlobalPool (Avg and Max)
        - [x] Upsample
        
        Other Extra Functionality:
        - [x] Flatten
        - [x] Activation
        - [x] Reshape

    - [x] Initializers
        - [x] Constant
        - [x] Uniform
        - [x] Normal
        - [x] Identity
        
    - [x] Losses
        - [x] MSE  (Mean Squared Error)
        - [x] MAE  (Mean Absolute Error)
        - [x] MAPE (Mean Absolute Percentage Error)
        - [x] BinaryCrossentropy
        - [x] CategoricalCrossentropy
        - [x] SparseCategoricalCrossentropy
        - [x] HuberLoss
        - [x] LogLossLinearActivation
        - [x] LogLossSigmoidActivation
        - [x] PerceptronCriterionLoss
        - [x] SvmHingeLoss

    - [x] Evaluation Metrics
        - [x] Accuracy
        - [x] TP (True Positives)
        - [x] TN (True Negatives)
        - [x] FP (False Positives)
        - [x] FN (False Negatives)
        - [x] Precision
        - [x] Recall
        - [x] F1Score
        
    - [ ] Regularizers
        - [x] L1
        - [x] L2
        - [ ] L1L2

    - [x] Optimization Modules for training
        - [x] SGD
        - [x] Momentum
        - [x] AdaGrad
        - [x] RMSProp
        - [x] AdaDelta
        - [x] Adam

    - [x] A Visualization Modules to track the training and testing processes
        - [x] History
        - [x] Verbosity

    - [x] A utils module for reading and saving models
    - [ ] Adding CUDA support
    - [ ] Publish to PyPI

### Example Usage:
- [ ] Download and Split a dataset (MNIST or CIFAR-10) to training, validation and testing
- [ ] Construct an Architecture [(LeNet or AlexNet)](https://engmrk.com/lenet-5-a-classic-cnn-architecture/) and make sure all of its components are provided in your framework.
    - [x] Conv2D Layer
    - [x] AvgPool Layer
    - [x] Flatten Layer
    - [x] Dense (Fully Connected) Layer
    - [x] Softmax Activation Function
- [ ] Train and test the model until a good accuracy is reached (Evaluation Metrics will need to be implemented in the framework also)
- [x] Save the model into a compressed format
