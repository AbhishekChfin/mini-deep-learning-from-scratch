# Mini Deep Learning Framework From Scratch

A compact neural network framework built in Python with NumPy to help understand both the software design and the mathematics behind deep learning.

To learn from this project in a meaningful way, implement the forward pass and backward pass yourself first, then compare your version with the code in this repository. If you want a step-by-step walkthrough of the core ideas, use [rough.ipynb](rough.ipynb) as a guided notebook.

## 1. What This Project Is

This project is a learning-focused deep learning framework built from scratch using Python and NumPy. Its purpose is not only to train models, but also to make the internals of neural networks visible: layers, parameters, losses, optimizers, and the flow of gradients.

The main goal is to understand how neural networks work from both a programming and mathematical perspective.

## 2. What Has Been Implemented

The framework currently includes a small set of core building blocks:

- Module base class
  - Defines a shared interface for forward pass, backward pass, parameter access, and gradient clearing.

- Parameter class
  - Stores trainable values and gradients, and provides simple gradient reset behavior.

- Linear layer
  - Implements a fully connected layer using $y = XW + b$.

- Activation layers
  - ReLU, Sigmoid, and Softmax are implemented as reusable layers that transform outputs and support gradient flow.

- MSELoss
  - Computes mean squared error and provides the basic gradient needed for regression-style learning.

- CrossEntropyLoss
  - Computes cross-entropy loss for probability-based targets.

- SoftmaxCrossEntropyLoss
  - Combines softmax and cross-entropy into a numerically stable training objective for classification.

- SGD optimizer
  - Updates parameters using gradient descent, with support for learning-rate decay, momentum, and Adagrad-style accumulation where implemented.

## 3. What You Can Learn From This Project

By reading and understanding the code, you can learn:

- Manual forward propagation through layers
- Manual backward propagation and gradient flow
- How losses initiate backpropagation
- How optimizers update parameters from gradients
- How layers, losses, optimizers, and training logic interact as a system
- How object-oriented programming makes neural network code modular, reusable, and easier to reason about

## 4. Python OOP Concepts Used

This project is a good example of how OOP helps structure neural network code.

- Abstraction: Module defines a common interface for forward, backward, parameters, and zero_grad.
- Inheritance: Linear and activation layers inherit from Module and reuse the same base behavior.
- Polymorphism: Different layers implement the same methods with different internal logic.
- Composition: Linear contains Parameter objects, and training code can compose layers into a custom computation graph.
- Encapsulation: Parameter owns its own data, gradient storage, and reset behavior.
- Separation of responsibilities: layers transform values, losses compute errors, and optimizers update parameters.

These concepts improve readability because they provide:

- consistent method names across components
- easier access to outputs, gradients, and parameters
- less duplicated logic
- a generic optimizer that can work with many parameter objects
- a modular way to build models from reusable pieces

## 5. Core Neural Network Mathematics

The code is built around a few essential mathematical ideas.

- Linear layer: $y = XW + b$
- ReLU: $f(x) = \max(0, x)$, with derivative 1 for $x > 0$ and 0 otherwise
- Sigmoid: $\sigma(x) = 1 / (1 + e^{-x})$, with derivative $\sigma(x)(1 - \sigma(x))$
- Softmax: $P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$
- MSE loss: $L = \text{mean}((y_{pred} - y_{true})^2)$, with gradient $\frac{2(y_{pred} - y_{true})}{N}$
- Cross entropy: $L = -\sum(y_{true} \log(y_{pred})) / batch\_size$
- Softmax + cross entropy: $\frac{\partial L}{\partial \text{logits}} = \frac{\text{softmax(logits)} - y_{true}}{batch\_size}$
- SGD: $parameter = parameter - learning\_rate \cdot gradient$
- Momentum: $update = momentum \cdot previous\_update - learning\_rate \cdot gradient$
- Adagrad: $cache = cache + gradient^2$, then $parameter = parameter - learning\_rate \cdot gradient / (\sqrt{cache} + \epsilon)$

## 6. Example Training Flow

A typical training step follows this order:

1. Forward pass through the layers
2. Compute the loss
3. Backward pass through the loss
4. Backward pass through the layers
5. Optimizer updates the parameters
6. Gradients are cleared for the next step

## 7. Project Goal / Roadmap

The next steps for this project are to make the framework more complete and practical:

- Dataset and DataLoader support
- a Trainer class
- an MNIST image classifier
- later transformer-style components such as embeddings, attention, and blocks

## 8. References

### Python OOP

- Abstraction: [https://medium.com/data-bistrot/abstraction-in-python-oop-c4da042f9eaf]
- Encapsulation: [https://medium.com/data-bistrot/understanding-encapsulation-in-object-oriented-programming-with-python-b7a65c994902]
- Inheritance: [https://medium.com/@rohanbhatotiya/what-is-inheritance-in-python-d2eaf4af87e2]
- Polymorphism: [https://medium.com/data-bistrot/polymorphism-in-python-object-oriented-programming-c652d8c3b792]
- Composition: [https://medium.com/data-bistrot/composition-vs-inheritance-in-python-oop-d4b3c3d8b463]
- Classes and Objects: [https://medium.com/@Shamimw/understanding-python-classes-objects-init-self-inheritance-polymorphism-encapsulation-f916ed99388e]

### Machine Learning

- Machine Learning Series by Dr. Raj Abhijit Dandekar: https://youtube.com/playlist?list=PLPTV0NXA_ZSj6tNyn_UadmUeU3Q3oR-hu&si=P6S7OSrTOkJyoTRE
- Dive into Deep Learning (D2L): https://d2l.ai/index.html


