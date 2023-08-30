import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


pathG = "/home/hmarefat/scratch/torchFOAM_archive/dynSmagCase/postProcessing/probes/500.096/grad(U)"
pathCs= "/home/hmarefat/scratch/torchFOAM_archive/dynSmagCase/postProcessing/probes/500/Cs"
pathUp= "/home/hmarefat/scratch/torchFOAM_archive/dynSmagCase/postProcessing/probes/500.096/UPrime2Mean"

nlns = 6104
skip = 877
nprb = 875

ncomG = 9 
G = np.zeros([nlns-skip, nprb, ncomG])
with open(pathG, "r") as f:
    for sk in range(skip):
        f.readline()
    for t in range(nlns-skip):    
        ls = f.readline().replace("(", "").replace(")", "").split()
        for prb in range(nprb):
            for c in range(ncomG):
                G[t][prb][c] = ls[prb*ncomG+c+1]
f.close()

#print(G[nlns-skip-1][nprb-1][ncomG-1])


ncomUp = 6 
Up = np.zeros([nlns-skip, nprb, ncomUp])
with open(pathUp, "r") as f:
    for sk in range(skip):
        f.readline()
    for t in range(nlns-skip):    
        ls = f.readline().replace("(", "").replace(")", "").split()
        for prb in range(nprb):
            for c in range(ncomUp):
                Up[t][prb][c] = ls[prb*ncomUp+c+1]
f.close()

#print(Up[0][0][-1])

Cs = np.zeros([nlns-skip, nprb])
with open(pathCs, "r") as f:
    for sk in range(skip):
        f.readline()
    for t in range(nlns-skip):    
        ls = f.readline().split()
        for prb in range(nprb):
            Cs[t][prb] = ls[prb+1]
f.close()

#print(Cs[-1][-1])

G_train = np.zeros([int(0.72*(nlns-skip))+1, nprb, ncomG])
Up_train = np.zeros([int(0.72*(nlns-skip))+1, nprb, ncomUp])
Cs_train = np.zeros([int(0.72*(nlns-skip))+1, nprb])

G_test = np.zeros([int(0.28*(nlns-skip)), nprb, ncomG])
Up_test = np.zeros([int(0.28*(nlns-skip)), nprb, ncomUp])
Cs_test = np.zeros([int(0.28*(nlns-skip)), nprb])

for t in range(int(0.54*(nlns-skip))+1):
    for p in range(nprb):
        for c in range(ncomG):
            G_train[t][p][c] = G[t][p][c]
        for c in range(ncomUp):
            Up_train[t][p][c] = Up[t][p][c]
        Cs_train[t][p] = Cs[t][p]

for t in range(int(0.82*(nlns-skip)), nlns-skip):
    for p in range(nprb):
        for c in range(ncomG):
            G_train[t-int(0.28*(nlns-skip))-1][p][c] = G[t][p][c]
        for c in range(ncomUp):
            Up_train[t-int(0.28*(nlns-skip))-1][p][c] = Up[t][p][c]
        Cs_train[t-int(0.28*(nlns-skip))-1][p] = Cs[t][p]
            
for t in range(int(0.54*(nlns-skip))+1, int(0.82*(nlns-skip))):
    for p in range(nprb):
        for c in range(ncomG):
            G_test[t-int(0.54*(nlns-skip))-1][p][c] = G[t][p][c]
        for c in range(ncomUp):
            Up_test[t-int(0.54*(nlns-skip))-1][p][c] = Up[t][p][c]
        Cs_test[t-int(0.54*(nlns-skip))-1][p] = Cs[t][p]
            
print('shapes:\n')
print(G.shape, '\t', Up.shape, '\t', Cs.shape)
print(G_train.shape, '\t', Up_train.shape, '\t', Cs_train.shape)
print(G_test.shape, '\t', Up_test.shape, '\t', Cs_test.shape)


nSamp_tr = (int(0.72*(nlns-skip))+1) * nprb
print(nSamp_tr)
nFeat = ncomG + ncomUp
nTarg = 1
data_tr = np.zeros([nSamp_tr, nFeat+nTarg])
feat_tr = np.zeros([nSamp_tr, nFeat])
targ_tr = np.zeros([nSamp_tr, nTarg])

nSamp_ts = int(0.28*(nlns-skip)) * nprb
data_ts = np.zeros([nSamp_ts, nFeat+nTarg])
feat_ts = np.zeros([nSamp_ts, nFeat])
targ_ts = np.zeros([nSamp_ts, nTarg])

count = 0
for p in range(nprb):
    for t in range(int(0.72*(nlns-skip))+1):
        for c in range(ncomG):
            feat_tr[count][c] = G_train[t][p][c]
            data_tr[count][c] = G_train[t][p][c]
        for c in range(ncomUp):
            feat_tr[count][c+ncomG] = Up_train[t][p][c]
            data_tr[count][c+ncomG] = Up_train[t][p][c]
        targ_tr[count] = Cs_train[t][p]
        data_tr[count][-1] = Cs_train[t][p]
        count = count + 1

count = 0
for p in range(nprb):
    for t in range(int(0.28*(nlns-skip))):
        for c in range(ncomG):
            feat_ts[count][c] = G_test[t][p][c]
            data_ts[count][c] = G_test[t][p][c]
        for c in range(ncomUp):
            feat_ts[count][c+ncomG] = Up_test[t][p][c]
            data_ts[count][c+ncomG] = Up_test[t][p][c]
        targ_ts[count] = Cs_test[t][p]
        data_ts[count][-1] = Cs_test[t][p]
        count = count + 1


sScaler = StandardScaler()
sScaler.fit(data_tr)
np.savetxt('./data_tr_means.txt',sScaler.mean_)
np.savetxt('./data_tr_scales.txt',sScaler.scale_)
data_tr_norm = sScaler.transform(data_tr)  

sScaler = StandardScaler()
sScaler.fit(data_ts)
np.savetxt('./data_ts_means.txt',sScaler.mean_)
np.savetxt('./data_ts_scales.txt',sScaler.scale_)
data_ts_norm = sScaler.transform(data_ts)
        
np.savetxt('./dynSmagProbes_train_norm.npy', data_tr_norm)
np.savetxt('./dynSmagProbes_test_norm.npy', data_ts_norm)
