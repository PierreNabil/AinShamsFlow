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

![UML Diagram of the Project]()

This is how the Design Structure should work in our Framework.

![Desgin Structure]()

## Usage:
You can start by importing the module as follows:
```python
import ainshamsflow as asf
```
Then you have access to all functionality in the modules as follows.
```python
asf.module_name.Class_name
```
example usage can be found in [main.py](https://github.com/PierreNabil/AinShamsFlow/blob/master/main.py)


## Team Members:
* [Pierre Nabil](https://github.com/PierreNabil)

* [Ahmed Taha]()

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
    - [ ] Activation Functions
        - [x] Linear
        - [ ] Sign
        - [x] Sigmoid
        - [ ] Hard Sigmoid
        - [x] Tanh
        - [ ] Hard Tanh
        - [x] ReLU
        - [x] LeakyReLU
        - [ ] ELU
        - [ ] SELU
        - [ ] Softmax
        - [ ] Softplus
        - [ ] Siftsign
        - [ ] Swish

    - [ ] Layers
        DNN Layers:
        - [x] Dense
        - [ ] BatchNorm
        - [ ] Dropout
        CNN Layers:
        - [ ] Conv (1D and 2D)
        - [ ] Pool (Avg and Max)
        - [ ] GlobalPool
        - [ ] Flatten
        - [ ] Upsample (1D and 2D)
        Other Extra Functionality:
        - [ ] Lambda
        - [ ] Activation
        - [ ] Add
        - [ ] Concatenate
        - [ ] Reshape

    - [ ] Initializers
        - [ ] Constant
        - [ ] Uniform
        - [ ] Normal
        - [ ] Identity
        
    - [ ] Losses
        - [x] MSE  (Mean Squared Error)
        - [ ] RMSE (Root Mean Squared Error)
        - [x] MAE  (Mean Absolute Error)
        - [x] MAPE (Mean Absolute Percentage Error)
        - [ ] BinaryCrossentropy
        - [ ] CategoricalCrossentropy
        - [ ] SparseCategoricalCrossentropy

    - [ ] Evaluation Metrics
        - [x] Accuracy
        - [ ] TP (True Positives)
        - [ ] TN (True Negatives)
        - [ ] FP (False Positives)
        - [ ] FN (False Negatives)
        - [ ] Precision
        - [ ] Recall
        - [ ] F1Score
        
    - [ ] Regularizers
        - [x] L1
        - [x] L2
        - [ ] L1L2

    - [ ] Optimization Modules for training
        - [x] SGD
        - [ ] Momentum
        - [ ] RMSProp
        - [ ] Adam

    - [ ] A Visualization Modules to track the training and testing processes
        - [x] History
        - [x] Verbosity
        - [ ] Something like TensorBoard? ...

    - [x] A utils module for reading and saving models
    - [ ] Adding CUDA support
    - [ ] Publish to PyPI

### Example Usage:
- [ ] Download and Split a dataset (MNIST or CIFAR-10) to training, validation and testing
- [ ] Construct an Architecture [(LeNet or AlexNet)](https://engmrk.com/lenet-5-a-classic-cnn-architecture/) and make sure all of its components are provided in your framework.
    - [ ] Conv2D Layer
    - [ ] AvgPool Layer
    - [ ] Flatten Layer
    - [x] Dense (Fully Connected) Layer
    - [ ] Softmax Activation Function
- [ ] Train and test the model until a good accuracy is reached (Evaluation Metrics will need to be implemented in the framework also)
- [x] Save the model into a compressed format
