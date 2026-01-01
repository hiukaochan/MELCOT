import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LRModel(nn.Module):
    def __init__(self, input_dim=10,  output_dim=10):
        super(LRModel, self).__init__()
        self.layer2 = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.sigmoid_ne = True

    def forward(self, x):
        if self.sigmoid_ne:
            x = self.sigmoid(self.layer2(x))
        else:
            x = self.layer2(x)
        return x