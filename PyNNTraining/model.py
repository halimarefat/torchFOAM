import torch

class nnModel(torch.nn.Module):

    def __init__(self, inp_s, hid_s, out_s, layer_num):
        super(nnModel, self).__init__()

        self.numl = layer_num
        self.inpl = torch.nn.Linear(inp_s, 12) #hid_s)
        self.actn = torch.nn.ReLU()
        self.hidl = torch.nn.Linear(12, 6) #hid_s, hid_s)
        self.outl = torch.nn.Linear(6, 1) #hid_s, out_s)

    def forward(self, x):
        x = self.inpl(x)
        x = self.actn(x)
        x = self.hidl(x)
        x = self.actn(x)
        #for _ in range(self.numl):
        #    x = self.hidl(x)
        #    x = self.actn(x)
        x = self.outl(x)
        return x

