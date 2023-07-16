import os
import torch
from torch.utils.data import Dataset
import math


class LatentsDataset(Dataset):

    def __init__(self, path):
        self.latents = torch.load(path,map_location=torch.device('cpu'))
        self.space = 'w+'
        self.latents.requires_grad = False

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, index):

        return self.latents[index]


class OneOfLatentsDataset(LatentsDataset):
    def __init__(self, path, index):
        super(OneOfLatentsDataset, self).__init__(path)
        self.index = index

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.latents[self.index]


class StyleSpaceLatentsDataset(Dataset):

    def __init__(self, path):
        self.latents = torch.load(path, map_location=torch.device('cpu'))
        self.space = 'style'

    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, index):
        return self.latents[index]


class NormalDataset(Dataset):

    def __init__(self, dim, length=10000):
        self.dim = dim
        self.length = length
        self.space = 'normal'

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        latent = torch.randn(
            1, self.dim, device=torch.device('cpu')
        )
        return latent