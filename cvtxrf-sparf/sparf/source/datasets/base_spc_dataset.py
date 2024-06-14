import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BaseSPCDataset(Dataset): 
    def __init__(self, opt): 
        super().__init__()
        self.opt = opt

    def __getitem__(self, idx):
        raise NotImplementedError