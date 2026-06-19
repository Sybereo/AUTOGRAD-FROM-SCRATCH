# My First Autograd (Micrograd Implementation)

A tiny, scalar-valued automatic differentiation engine with a dynamic computational graph wrapper, built completely from scratch in Python. This library implements backpropagation (reverse-mode autodiff) over a custom graph structure, capable of building and training Deep Multi-Layer Perceptrons (MLPs).

## 🚀 Features

* **Dynamic Graph Construction:** Tracks mathematical operations dynamically using pointer-based dependency trees.
* **Automated Backpropagation:** Implements the chain rule automatically via recursive topological sorting.
* **Modular Neural Network Components:** Includes object-oriented abstractions for individual `Neurons`, full `Layers`, and sequential `MLPs`.
* **Standard Optimizations:** Features generalized type-fallback operators (`__radd__`, `__rsub__`) to support fluid cross-operations between custom nodes and standard numeric integers/floats.

## 🛠️ Tech Architecture

* **Language:** Python 3
* **Core Framework:** Pure Python (Zero external dependencies for the math engine)
* **Mathematical Operations Supported:** Addition, Subtraction, Multiplication, Negation, Powers, and `tanh` activation functions.

## 📦 Usage Example

Since all components are built from scratch, they run natively within the notebook cells without external dependencies:

```python
# 1. Initialize a neural network (3 inputs, two hidden layers of 4, 1 output)
n = MLP(3, [4, 4, 1])

# 2. Define a batch of inputs and ground truth targets
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

# 3. Training Loop (20 Epochs)
for k in range(20):
  # foward pass
  ypred = [n(x) for x in xs]
  
  # implementing mean squared error loss
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

  # backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()

  # update
  for p in n.parameters():
    p.data += -0.05 * p.grad

  print(k, loss.data)
