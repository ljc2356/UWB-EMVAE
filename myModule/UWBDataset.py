import torch
import numpy as np
from torch.utils.data import Dataset

class UWBDataset(Dataset):
    def __init__(self,data):
        self.data = data
        self.data_num = self.data.shape[0]
    def __getitem__(self, item):
        csi = torch.Tensor(self.data[item,0:2,0:8,0:50]) #(N,2,8,50)
        x_index_1 = torch.Tensor(np.array(self.data[item,0,0,51])) - 1 #(N,1)
        a_1 = torch.Tensor(self.data[item,0:2,0:8,50]).unsqueeze(dim=2)  #(N,2,8,1)
        x_index_2 = torch.Tensor(np.array(self.data[item,0,0,53])) - 1 #(N,1)
        a_2 = torch.Tensor(self.data[item,0:2,0:8,52]).unsqueeze(dim=2)  #(N,2,8,1)
        return csi,x_index_1,a_1,x_index_2,a_2

    def __len__(self):
        return self.data_num
