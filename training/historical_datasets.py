from pydantic import BaseModel
from typing import Optional, Callable
import torch
from torch.utils.data import Dataset


class HistoricalDatasetConfig(BaseModel):
    num_episodes: int = 1000
    episode_length: int = 100
    num_classes: int = 2
    num_features: int = 10
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    class Config:
        arbitrary_types_allowed = True


class HistoricalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, config: HistoricalDatasetConfig):
        """
        Args:
            config (HistoricalDatasetConfig): Configuration object for dataset parameters.
        """
        self.episode_length = config.episode_length
        self.num_episodes = config.num_episodes
        self.num_classes = config.num_classes
        self.num_features = config.num_features
        self.transform = config.transform
        self.data = self.generate_data()

    def generate_data(self):
        features = torch.randn(
            self.num_episodes, self.episode_length, self.num_features
        )
        labels = torch.randint(
            0, self.num_classes, (self.num_episodes, self.episode_length)
        )
        return features, labels

    def __len__(self):
        return self.num_episodes * self.episode_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract features and label for a single episode
        episode_idx = idx // self.episode_length
        time_idx = idx % self.episode_length
        features = self.data[0][episode_idx, time_idx]
        label = self.data[1][episode_idx, time_idx]
        if self.transform:
            features = self.transform(features)
        return features, label


class ProbIdentityDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, config: HistoricalDatasetConfig):
        """
        he        Args:
                    config (HistoricalDatasetConfig): Configuration object for dataset parameters.
        """

        self.length = config.episode_length * config.num_episodes
        self.num_classes = config.num_classes
        self.num_features = config.num_features
        self.transform = config.transform
        self.data = self.generate_data()

    def generate_data(self):
        features = torch.randn(self.length, self.num_classes)
        raw_labels = (torch.randn(self.length, self.num_classes) < features).to(
            torch.float32
        )
        labels = torch.nn.functional.softmax(raw_labels, dim=1)
        return features, labels

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract features and label for a single episode
        features = self.data[0][idx]
        label = self.data[1][idx]
        if self.transform:
            features = self.transform(features)
        return features, label
