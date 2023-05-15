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


data = CustomDataset('/home/hmarefat/scratch/torchFOAM/datasetGen/dataset_scaled.npy')

train_val_split = 0.8
batch_sz = 1028
inp_sz = 9
out_sz = 1
hid_sz = 63
lay_nm = 4
device = torch.device("cuda")
epochs = 600
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
print('+-- dataset  shape: [', len(data)    ,',', inp_sz+out_sz, ']')
print('+-- train ds shape: [', len(train_ds),',', inp_sz+out_sz, ']')
print('+-- val   ds shape: [', len(val_ds)  ,',', inp_sz+out_sz, ']')

train_noBatch = int(len(train_ds) / batch_sz)
val_noBatch = int(len(val_ds) / batch_sz)

print('+-- train batchs #: ', train_noBatch)
print('+-- val   batchs #: ', val_noBatch)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



"""
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
    model.train()
    Loss_train = 0 
    Loss_val = 0
    for i, batch in enumerate(train_loader):
        train_feat = batch["data"].to(device).to(torch.float32)
        train_labs = batch["target"].to(device).to(torch.float32)
        train_pred = model.forward(train_feat)
        train_loss = torch.nn.functional.mse_loss(train_pred, train_labs)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        Loss_train += train_loss.item()
    Loss_train /= batch_sz

    model.eval()
    batch_indx = 0
    for batch in val_loader:
        val_feat = batch["data"].to(device).to(torch.float32)
        val_labs = batch["target"].to(device).to(torch.float32)
        val_pred = model.forward(val_feat)
        val_loss = torch.nn.functional.mse_loss(val_pred, val_labs)

        optimizer.zero_grad()
        val_loss.backward()
        optimizer.step()

        Loss_val += val_loss.item()
        batch_indx += 1
    Loss_val /= batch_sz
    print(f"Epoch: {epoch} / {epochs}, Train Loss: {Loss_train}, Val Loss: {Loss_val}.")

