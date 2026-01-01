import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class DNNModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=10, output_dim=10, dropout_rate=0):
        super(DNNModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.sigmoid_ne = True

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        if self.sigmoid_ne:
            x = self.sigmoid(self.layer2(x))
        else:
            x = self.layer2(x)
        return x
