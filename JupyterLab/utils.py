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

def returnRandmIndx(maxIndx, count):
    return np.array(random.sample(range(1,maxIndx), count))

def splitterIndx(indx):
    seen = indx[:int(0.7*indx.shape[0])]
    unseen = indx[int(0.7*indx.shape[0]):]
    
    return seen, unseen

def scaler(name, ds):
    scaler = StandardScaler()
    scaler.fit(ds)
    np.savetxt(f'../processedDatasets/{name}_means.txt',scaler.mean_)
    np.savetxt(f'../processedDatasets/{name}_scales.txt',scaler.scale_)
    ds_norm = scaler.transform(ds)   
    np.savetxt(f'../processedDatasets/{name}_norm.txt', ds_norm, fmt='%.18e')
    np.savetxt(f'../processedDatasets/{name}.txt', ds, fmt='%.18e')
    
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


