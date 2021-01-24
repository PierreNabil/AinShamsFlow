# AinShamsFlow
4th CSE Neural Networks Project.

![asf_logo](/images/asf_logo.png)

## Contents:
* [Project Description](#Project-Description)
* [Project Structure](#Project-Structure)
* [Install](#Install)
* [Usage](#Usage)
* [Team Members](#Team-Members)
* [Todo](#Todo)

## Project Description:
AinShamsFlow (asf) is a Deep Learning Framework built by our [Team](#Team-Members)
from Ain Shams University during December 2020 - January 2021.

The Design and Interface is inspired heavily from Keras and TensorFlow.

However, we only implement everything from scratch using only simple libraries
such as numpy and matplotlib.

## Project Structure:
The Design of all parts can be seen in the following UML Diagram.

![UML Diagram of the Project](/images/UML%20Class%20Diagram.png)

This is how the Design Structure should work in our Framework.

![Desgin Structure](/images/Design%20Structure.png)

## Install:
You can install the latest available version as follows:
```shell
$ pip install ainshamsflow
```

## Usage:
you start using this project by importing the package as follows:

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


A more elaborate example usage can be found in [main.py](/main.py) or check out this 
[demo notebook](https://colab.research.google.com/drive/1sqEeUUkG3bTplhlLb73QCGUbaubX2aXi?usp=sharing).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)
](https://colab.research.google.com/drive/1sqEeUUkG3bTplhlLb73QCGUbaubX2aXi?usp=sharing)

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
- [x] A Data Module to read and process datasets.

    - [x] Dataset
    
        - [x] \_\_init\_\_()
        - [x] \_\_bool\_\_()
        - [x] \_\_len\_\_()
        - [x] \_\_iter\_\_()
        - [x] \_\_next\_\_()
        - [x] apply()
        - [x] numpy()
        - [x] batch()
        - [x] cardinality()
        - [x] concatenate()
        - [x] copy()
        - [x] enumerate()
        - [x] filter()
        - [x] map()
        - [x] range()
        - [x] shuffle()
        - [x] skip()
        - [x] split()
        - [x] take()
        - [x] unbatch()
        - [x] add_data()
        - [x] add_targets()
        - [x] normalize()
        
    - [x] ImageDataGenerator
    
        - [x] \_\_init\_\_()
        - [x] flow_from_directory()

- [x] A NN Module to design different architectures.

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
    
        - DNN Layers:
            - [x] Dense
            - [x] BatchNorm
            - [x] Dropout
        
        - CNN Layers 1D: (optional)
            - [x] Conv
            - [x] Pool (Avg and Max)
            - [x] GlobalPool (Avg and Max)
            - [x] Upsample
        
        - CNN Layers 2D:
            - [x] Conv
            - [x] Pool (Avg and Max)
            - [x] FastConv
            - [x] FastPool (Avg and Max)
            - [x] GlobalPool (Avg and Max)
            - [x] Upsample
        
        - CNN Layers 3D: (optional)
            - [x] Conv
            - [x] Pool (Avg and Max)
            - [x] GlobalPool (Avg and Max)
            - [x] Upsample
        
        - Other Extra Functionality:
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
        
    - [x] Regularizers
    
        - [x] L1
        - [x] L2
        - [x] L1_L2

    - [x] Optimization Modules for training
    
        - [x] SGD
        - [x] Momentum
        - [x] AdaGrad
        - [x] RMSProp
        - [x] AdaDelta
        - [x] Adam

    - [x] A Visualization Modules to track the training and testing processes
    
        - [x] History Class for showing training statistics
        - [x] ```verbose``` parameter in training
        - [x] live plotting of training statistics

    - [x] A utils module for reading and saving models
    - [ ] Adding CUDA support
    - [x] Publish to PyPI
    - [x] Creating a Documentation for the Project

### Example Usage:
This part can be found in the
[demo notebook](https://colab.research.google.com/drive/1sqEeUUkG3bTplhlLb73QCGUbaubX2aXi?usp=sharing)
mentioned above.

- [x] Download and Split a dataset (MNIST or CIFAR-10) to training, validation and testing
- [x] Construct an Architecture ([LeNet](https://engmrk.com/lenet-5-a-classic-cnn-architecture/) 
or [AlexNet](https://dl.acm.org/doi/abs/10.1145/3065386)) and make sure all of its components are 
provided in your framework.
- [x] Train and test the model until a good accuracy is reached (Evaluation Metrics will need to be implemented in the framework also)
- [x] Save the model into a compressed format
