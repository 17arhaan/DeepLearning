import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from torch import nn
from torch.utils.data import DataLoader , Dataset
from sklearn.model_selection import train_test_split




actual = [1.0 , 2.0 , 3.0 , 4.0 , 5.0 , 6.0 , 7.0 , 8.0 , 9.0 , 10.0]
predicted = [1.2 , 2.1 , 2.9 , 4.0 , 5.2 , 6.1 , 7.0 , 8.1 , 9.0 , 10.1]
plt.plot(actual)
plt.plot(predicted)
plt.xticks([])
plt.yticks([])
plt.show()