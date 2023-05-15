import torch
from tqdm import tqdm
import numpy as np
from model import nnModel

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = np.loadtxt(path) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx][0:-1]
        labels = self.data[idx][-1]
        sample = {"data": data, "target": np.expand_dims(labels, axis=0)}
        return sample


data = CustomDataset('/home/hmarefat/scratch/torchFOAM/datasetGen/dataset_460_490.npy')

train_val_split = 0.8
batch_sz = 1028
inp_sz = 9
out_sz = 1
hid_sz = 63
lay_nm = 4
device = torch.device("cuda")
epochs = 50
train_sz = int(train_val_split * len(data))
val_sz = len(data) - train_sz

train_ds, val_ds = torch.utils.data.random_split(data, [train_sz, val_sz]) 

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_sz, shuffle=True,
                                           pin_memory=True, drop_last=True)
val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=batch_sz, shuffle=True,
                                           pin_memory=True, drop_last=True)


model = nnModel(inp_s=inp_sz, hid_s=hid_sz, out_s=out_sz, layer_num=lay_nm)
model.to(device)

print('+-- model summary:')
print(model)
print('+-- dataset  shape: [', len(data), )
print('+-- train ds shape: [', len(train_ds))
print('+-- val   ds shape: [', len(val_ds))

train_noBatch = len(train_ds) / batch_sz
val_noBatch = len(val_ds) / batch_sz

print('+-- train batchs #: ', train_noBatch)
print('+-- val   batchs #: ', val_noBatch)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

Loss_train = 0 
model.train()
model.train() 
for epoch in range(epochs):
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            
            optimizer.zero_grad()
            train_feat = batch["data"].to(device).to(torch.float32)
            train_labs = batch["target"].to(device).to(torch.float32)
            train_pred = model.forward(train_feat)
            train_loss = torch.nn.functional.mse_loss(train_pred, train_labs) 
            train_accy = (train_pred == train_labs).sum().item() / batch_sz
            train_loss.backward()
            optimizer.step()

    Loss_train += train_loss.item() / batch_sz

    tepoch.set_postfix(loss=Loss_train, accuracy=100. * train_accy)
"""
for epoch in range(epochs):
    print(epoch)
    for batch in train_loader:
        train_feat = batch["data"].to(device).to(torch.float32)
        train_labs = batch["target"].to(device).to(torch.float32)
        train_pred = model.forward(train_feat)
        train_loss = torch.nn.functional.mse_loss(train_pred, train_labs)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        Loss_train += train_loss.item()
    Loss_train /= 1028 

#torch.jit.load(model, "/home/hmarefat/scratch/torchFOAM/nnTraining/best_model_0510235.pt")
"""
