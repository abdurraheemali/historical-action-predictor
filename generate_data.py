# generate_data.py
import torch
from datasets import HistoricalDataset
import os

def main():
    # Create a HistoricalDataset with 1000 episodes, each of length 100
    dataset = HistoricalDataset(num_episodes=1000, episode_length=100)

    # Print the length of the dataset
    print(f"Dataset length: {len(dataset)}")

    # Get and print the first item in the dataset
    first_item = dataset[0]
    print(f"First item in dataset: {first_item}")

    # Ensure the data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save the features and labels separately to the data directory
    torch.save([dataset[i] for i in range(len(dataset))], os.path.join('data', 'generated_dataset.pt'))

if __name__ == "__main__":
    main()