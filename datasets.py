from pydantic import BaseModel
import torch
from torch.utils.data import Dataset, random_split


class HistoricalDatasetConfig(BaseModel):
    num_episodes: int = 1000
    episode_length: int = 100
    num_classes: int = 2
    transform: callable = None


class HistoricalDataset(Dataset):
    def __init__(self, config: HistoricalDatasetConfig):
        """
        Args:
            config (HistoricalDatasetConfig): Configuration object for dataset parameters.
        """
        self.episode_length = config.episode_length
        self.num_episodes = config.num_episodes
        self.num_classes = config.num_classes
        self.transform = config.transform
        self.data = self.generate_data()

    def generate_data(self):
        # Initialize tensors to hold the features and labels
        features = torch.randn(self.num_episodes, self.episode_length)
        labels = torch.randint(0, self.num_classes, (self.num_episodes,))
        return features, labels

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        # Extract features and label for a single episode
        features = self.data[0][idx]
        label = self.data[1][idx]
        if self.transform:
            features = self.transform(features)
        return features, label
