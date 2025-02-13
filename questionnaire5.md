
### **Deep Learning Lab Viva Questionnaire (Till Lab 5)**

#### **Lab 1: Introduction to Tensors**
1. **What is a tensor in PyTorch?**  
   - A tensor is a multi-dimensional array similar to NumPy arrays but with GPU support.

2. **How can you create a tensor in PyTorch? Provide an example.**  
   ```python
   import torch
   tensor = torch.tensor([1, 2, 3])
   print(tensor)
   ```

3. **What is the difference between a tensor and a NumPy array?**  
   - Tensors can run on GPUs, while NumPy arrays are CPU-bound.

4. **How can you convert a NumPy array to a tensor?**  
   ```python
   import numpy as np
   np_array = np.array([1, 2, 3])
   tensor = torch.tensor(np_array)
   ```

5. **Explain the use of `.item()` method in PyTorch.**  
   - It extracts a Python scalar from a tensor with a single value.

---

#### **Lab 2: Computational Graphs**
6. **What is a computational graph in deep learning?**  
   - It is a directed acyclic graph where nodes represent operations and edges represent tensors.

7. **What is the role of `requires_grad=True` in PyTorch tensors?**  
   - It enables automatic differentiation for tensor computations.

8. **What is backpropagation?**  
   - It is the process of computing gradients to update model weights.

9. **How can you compute gradients in PyTorch? Provide an example.**  
   ```python
   x = torch.tensor(3.0, requires_grad=True)
   y = x ** 2
   y.backward()
   print(x.grad)  # Output: 6.0
   ```

10. **What is the difference between `.detach()` and `.requires_grad_(False)`?**  
    - `.detach()` returns a tensor that does not track gradients, whereas `.requires_grad_(False)` modifies the tensor in-place.

---

#### **Lab 3: Deep Learning Library in PyTorch**
11. **What is PyTorch?**  
    - PyTorch is an open-source deep learning library for tensor computation and automatic differentiation.

12. **What is the purpose of `torch.nn.Module` in PyTorch?**  
    - It is the base class for all neural network models.

13. **Write a simple PyTorch implementation for Linear Regression.**
   ```python
   import torch.nn as nn
   class LinearRegression(nn.Module):
       def __init__(self):
           super().__init__()
           self.linear = nn.Linear(1, 1)
       def forward(self, x):
           return self.linear(x)
   ```

14. **What is the role of `torch.optim` in PyTorch?**  
    - It provides optimization algorithms like SGD, Adam for training neural networks.

15. **How do you define a loss function in PyTorch?**  
   ```python
   loss_fn = nn.MSELoss()
   ```

---

#### **Lab 4: Feed-Forward Neural Networks**
16. **What is a feed-forward neural network?**  
    - It is a neural network where connections do not form a cycle.

17. **List the three main layers in a feed-forward neural network.**  
    - Input Layer, Hidden Layers, Output Layer.

18. **What is the role of activation functions in neural networks?**  
    - They introduce non-linearity, enabling the network to learn complex patterns.

19. **What is the difference between ReLU and Sigmoid activation functions?**  
    - ReLU allows non-negative values only, while Sigmoid maps values to (0,1).

20. **Write a PyTorch implementation for a simple feed-forward network.**
   ```python
   class NeuralNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(10, 5)
           self.relu = nn.ReLU()
           self.fc2 = nn.Linear(5, 1)
       def forward(self, x):
           x = self.relu(self.fc1(x))
           return self.fc2(x)
   ```

---

#### **Lab 5: Convolutional Neural Networks (CNNs)**
21. **What is a Convolutional Neural Network (CNN)?**  
    - A deep learning model designed for image processing, using convolutional layers.

22. **Why do we use convolution instead of a fully connected network for images?**  
    - It preserves spatial relationships and reduces the number of parameters.

23. **What is the role of a kernel in a CNN?**  
    - A kernel is a small matrix used for feature detection through convolution.

24. **How does max-pooling help in CNNs?**  
    - It reduces spatial dimensions while retaining important features.

25. **Write a PyTorch implementation for a simple CNN.**
   ```python
   class CNN(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
           self.pool = nn.MaxPool2d(2, 2)
           self.fc1 = nn.Linear(32 * 13 * 13, 10)
       def forward(self, x):
           x = self.pool(F.relu(self.conv1(x)))
           x = x.view(-1, 32 * 13 * 13)
           return self.fc1(x)
   ```

---
