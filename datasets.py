import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np 

class HistoricalDataset(Dataset):
    def __init__(self, num_episodes=1000, episode_length=100, transform=None):
        """
        Args:
            num_episodes (int): Number of episodes to generate
            episode_length (int): Length of each episode
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = self.generate_data(num_episodes, episode_length)
        self.transform = transform

    def generate_data(self, num_episodes, episode_length):
        # Initialize an empty list to hold the episodes
        data = []

        # Generate the episodes
        for _ in range(num_episodes):
            # Assuming the last element is the label
            episode = np.random.randn(episode_length + 1)
            features = episode[:-1]
            label = episode[-1]
            data.append((features, label))

        return data

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Extract a single item from the dataset
        sample = self.data[idx]
        if self.transform:
            sample = (self.transform(sample[0]), sample[1])
        return sample