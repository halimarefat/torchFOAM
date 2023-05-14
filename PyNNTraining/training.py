import torch
import numpy as np
from model import nnModel


data = np.loadtxt('/home/hmarefat/scratch/torchFOAM/datasetGen/dataset_460_490.npy')

train_loader = torch.utils.data.DataLoader(data, batch_size=1028, shuffle=False)

model = nnModel(inp_s=9, hid_s=63, out_s=1, layer_num=4)

torch.jit.load(model, "/home/hmarefat/scratch/torchFOAM/nnTraining/best_model_0510235.pt")
print(model)
print(data.shape)