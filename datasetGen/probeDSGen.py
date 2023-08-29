import numpy as np
import pandas as pd


pathG = "/home/hmarefat/scratch/torchFOAM/dynSmagCase/postProcessing/probes/500.096/grad(U)"
pathCs= "/home/hmarefat/scratch/torchFOAM/dynSmagCase/postProcessing/probes/500/Cs"
pathUp= "/home/hmarefat/scratch/torchFOAM/dynSmagCase/postProcessing/probes/500.096/UPrime2Mean"

nlns = 6104
skip = 877
nprb = 875


ncom = 9 
G = np.zeros([nlns-skip, nprb, ncom])
with open(pathG, "r") as f:
    for sk in range(skip):
        f.readline()
    for t in range(nlns-skip):    
        ls = f.readline().replace("(", "").replace(")", "").split()
        for prb in range(nprb):
            for c in range(ncom):
                #print(t+1, prb+1, c+1)
                G[t][prb][c] = ls[prb*ncom+c+1]
f.close()

print(G[nlns-skip-1][nprb-1][ncom-1])

ncom = 6 
Up = np.zeros([nlns-skip, nprb, ncom])
with open(pathUp, "r") as f:
    for sk in range(skip):
        f.readline()
    for t in range(nlns-skip):    
        ls = f.readline().replace("(", "").replace(")", "").split()
        for prb in range(nprb):
            for c in range(ncom):
                #print(t+1, prb+1, c+1)
                Up[t][prb][c] = ls[prb*ncom+c+1]
f.close()

print(Up[nlns-skip-1][nprb-1][-2])

Cs = np.zeros([nlns-skip, nprb])
with open(pathCs, "r") as f:
    for sk in range(skip):
        f.readline()
    for t in range(nlns-skip):    
        ls = f.readline().split()
        for prb in range(nprb):
            Cs[t][prb] = ls[prb+1]
f.close()

print(Cs[-1][-1])