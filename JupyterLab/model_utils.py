import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import sys
import random
from sklearn.preprocessing import StandardScaler
from IPython.display import display, Math
from torch.utils.data import DataLoader, Dataset

sys_epsilon = sys.float_info.epsilon

class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __getitem__(self, index):
        # Ensure all indices are valid
        if index < len(self.data):
            return torch.tensor(self.data.iloc[index].values, dtype=torch.float64)
        else:
            raise IndexError(f"Index {index} out of range for dataset with length {len(self.data)}")

    def __len__(self):
        return len(self.data)
    
def coeff_determination(y_true, y_pred):
    SS_res = torch.sum(torch.square( y_true - y_pred ))
    SS_tot = torch.sum(torch.square( y_true - torch.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + sys_epsilon) )

class MLPModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer):
        super(MLPModel, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, neurons_per_layer[0]))
        layers.append(nn.ReLU())

        for i in range(1, hidden_layers):
            layers.append(nn.Linear(neurons_per_layer[i - 1], neurons_per_layer[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(neurons_per_layer[-1], output_size))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x
    
class EarlyStopper:
    def __init__(self, patience=1, path=None):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.min_val_loss = float('inf')

    def early_stop(self, model_stat, val_loss):
        if val_loss < self.min_val_loss:
            torch.save(model_stat, self.path)
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + sys_epsilon):
            self.counter += 1
            if self.counter >= self.patience:
                print('+++ Early Stopping is reached! +++')
                return True
        return False
    
def log_epoch_info(epoch, epochs, Loss_train, coeff_train, Loss_val, coeff_val):
    message = (
        f"Epoch: {epoch} / {epochs}, \n"
        f"Train -- Loss: {Loss_train}, Coeff: {coeff_train} \n"
        f"Val   -- Loss: {Loss_val}, Coeff: {coeff_val} \n\n"
    )
    print(message)