import numpy as np
from sklearn.preprocessing import StandardScaler

ds_1 = np.loadtxt('dynSmagCase_600_1200.npy')
ds_2 = np.loadtxt('_dynSmagCase_Re3e3_600_1200.npy')

file_mean = ['dynSmagCase_600_1200_means.txt', '_dynSmagCase_Re3e3_600_1200_means.txt']
file_scales = ['dynSmagCase_600_1200_scales.txt', '_dynSmagCase_Re3e3_600_1200_scales.txt']

mean = np.zeros((13,2))
stds = np.zeros((13,2))

for j in range(len(file_mean)):
    with open(file_mean[j], "r") as fin_mean:
        with open(file_scales[j], "r") as fin_stds:
            for i in range(len(mean)):
                mean[i][j] = float(fin_mean.readline())
                stds[i][j] = float(fin_stds.readline())
            
for i in range(len(ds_1)):
    for j in range(len(mean)):
        ds_1[i][j] = ds_1[i][j] * stds[j][0] + mean[j][0]

for i in range(len(ds_2)):
    for j in range(len(mean)):
        ds_2[i][j] = ds_2[i][j] * stds[j][1] + mean[j][1]    
    
ds = np.concatenate((ds_1, ds_2))

name = 'ds_UPSij_Re1e3_Re3e3_600_1200'

ds_scaler = StandardScaler()
ds_scaler.fit(ds)
np.savetxt('./'+name+'_means.txt',ds_scaler.mean_)
np.savetxt('./'+name+'_scales.txt',ds_scaler.scale_)
scaled_ds = ds_scaler.transform(ds)  

print('+-- mixed ds shape    : ', scaled_ds.shape)
np.savetxt(name+'.npy', scaled_ds)