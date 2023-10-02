import numpy as np
import torch.nn.functional as f
import torch
from torch.utils.data import Dataset
from splearn.datasets.base import load_data_sample

class PautomacDataset(Dataset):
    def __init__(self, label_path, data_path):
        self.labels = torch.Tensor(np.load(label_path)[:,1:,:])
        loaded_data = load_data_sample(data_path, filetype='Pautomac')
        self.data_tensor = torch.LongTensor(loaded_data.data) + 1 #[N, seq_length]
        self.nbL = loaded_data.nbL
        self.nbQ = self.labels.shape[2]
        self.T = self.data_tensor.shape[1]

    def __len__(self):
        return self.data_tensor.shape[0] 
    def __getitem__(self, idx):
        return self.data_tensor[idx], self.labels[idx]

class SyntheticDataset(Dataset):
    def __init__(self, label_path, data_path):
        self.labels = torch.Tensor(np.load(label_path)[:, 1:, :])
        self.data_tensor = torch.Tensor(np.load(data_path)[:,:,None])# [N, seq_length]
        self.nbL = int(torch.max(self.data_tensor))
        self.nbQ = self.labels.shape[2]
        self.T = self.data_tensor.shape[1]

    def __len__(self):
        return self.data_tensor.shape[0]

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.labels[idx]