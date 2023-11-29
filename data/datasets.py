import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np 

class HistoricalDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (string): Path to the data file
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = [] #TODO: copy format of mage model. 
        # Gotta be a context, action pair.

    def __len__(self):
        
        return len(self.data)
        
    def __getitem__(self, idx):
        # Extract a single item from the dataset
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample