import os
import numpy as np
from datasetGen import datasetGen

np.random.seed(10)

ds = datasetGen(caseFolder='dynSmagCase', lowbnd=525, highbnd=600, bounded=True) 
ds.__call__()

"""
np.random.shuffle(ds)

## Hyperparameters
in_channels = 9
out_channels = 1
num_layers = 6
num_neurons = 32

training_in = ds[:,0:in_channels]
training_out = ds[:,in_channels:]

print(f"The shape of training_in is {training_in.shape}")
print(f"The shape of training_out is {training_out.shape}")
"""
