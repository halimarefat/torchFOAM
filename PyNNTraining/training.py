import torch
import tqdm
import numpy as np
from model import nnModel

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = np.loadtxt(path) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx][0:-1]
        labels = np.expand_dims(self.data[idx][-1], axis=1)
        sample = {"data": data, "target": labels}
        return sample


data = CustomDataset('/home/hmarefat/scratch/torchFOAM/datasetGen/dataset_460_490.npy')

train_val_split = 0.8
batch_sz = 1028
device = torch.device("cuda")
epochs = 500
train_sz = int(train_val_split * len(data))
val_sz = len(data) - train_sz

train_ds, val_ds = torch.utils.data.random_split(data, [train_sz, val_sz]) 

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_sz, shuffle=True)


model = nnModel(inp_s=9, hid_s=63, out_s=1, layer_num=4)
model.to(device)

print('+-- model summary:')
print(model)
print('+-- dataset  shape: ', len(data))
#print('+-- train ds shape: ', train_ds.shape)
#print('+-- val   ds shape: ', val_ds.shape)

#train_noBatch = train_ds.shape[0] / batch_sz
#val_noBatch = val_ds.shape[0] / batch_sz

#print('+-- train batchs #: ', train_noBatch)
#print('+-- val   batchs #: ', val_noBatch)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

Loss_train = 0 
model.train()
for epoch in range(epochs):
    print(epoch)
    for batch in train_loader:
        train_feat = batch["data"].to(device).to(torch.float32)
        train_labs = batch["target"].to(device).to(torch.float32)
        train_pred = model.forward(train_feat)
        print(train_labs.size(), '\n', train_pred.size())
        train_loss = torch.nn.functional.mse_loss(train_pred, train_labs)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        Loss_train += train_loss.item()
    Loss_train /= 1028 

#torch.jit.load(model, "/home/hmarefat/scratch/torchFOAM/nnTraining/best_model_0510235.pt")

