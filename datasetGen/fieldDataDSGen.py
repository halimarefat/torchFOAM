import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


path = "/home/hmarefat/scratch/torchFOAM/dynSmagCase_2/postProcessing/fieldData.dat"
num = 2048000 #1048580
col = 10

data = np.zeros([num, col])
with open(path, "r") as f:
    f.readline()
    for i in range(num):
        l = f.readline().split()
        for j in range(col):
            data[i,j] = l[j] 
        
ds_scaler = StandardScaler()
ds_scaler.fit(data)
np.savetxt('./dynSmagData2_means.txt',ds_scaler.mean_)
np.savetxt('./dynSmagData2_scales.txt',ds_scaler.scale_)
data_norm = ds_scaler.transform(data)   

np.savetxt('dynSmagData2.txt', data_norm)