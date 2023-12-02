import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import HistoricalDataset
from torch.utils.data import  TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import HistoricalDatasetConfig, HistoricalDataset
import os
import random


# Set the random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# Determine the device to use
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

class ActionPredictor(nn.Module):
    def __init__(self, num_classes):
        super(ActionPredictor, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)


    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

config = HistoricalDatasetConfig(
    num_episodes=1000,
    episode_length=100,
    num_classes=2, 
    transform=None  
)

# Initialize models, optimizers, and other components
def initialize_components(model_class, num_classes, learning_rate=0.01, momentum=0.9):
    model = model_class(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    return model, criterion, optimizer, scheduler

# Instantiate the HistoricalDataset with the config
full_dataset = HistoricalDataset(config=config)
# Split the dataset into training and validation sets
num_train = int((1 - validation_split) * len(full_dataset))
num_val = len(full_dataset) - num_train
train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

# Create DataLoaders for the training and validation sets
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Move tensors to the chosen device
features_tensor = features_tensor.to(device)
labels_tensor = labels_tensor.to(device)

# Create a TensorDataset from the tensors
historical_dataset = TensorDataset(features_tensor, labels_tensor)

# Create a DataLoader from the TensorDataset
trainloader = DataLoader(historical_dataset, batch_size=64, shuffle=True)


for epoch in range(5):
    running_loss = 0.0

    performative_model.train() 

    for inputs, actions in trainloader:
        optimizer.zero_grad()
        outputs = performative_model(inputs) 
        loss = criterion(outputs, actions)  # Use raw logits and class indices for loss calculation
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

     val_loss = 0.0
    with torch.no_grad():
    for val_inputs, val_actions in valloader:
        val_outputs = performative_model(val_inputs)  # Changed 'model' to 'performative_model'
        val_loss += criterion(val_outputs, val_actions).item()
    val_loss /= len(valloader)

    # Check for early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(performative_model.state_dict(), os.path.join('results', 'models', 'best_performative_model.pth'))
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

    # Step the scheduler with the validation loss
    scheduler.step(val_loss)

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')
    # Save the model after each epoch
    torch.save(performative_model.state_dict(), os.path.join('results', 'models', f'performative_model_epoch_{epoch+1}.pth'))

# Zero Sum Predictor
zerosum_model_1 = ActionPredictor()
zerosum_model_2 = ActionPredictor()

# Move models to the chosen device
zerosum_model_1 = zerosum_model_1.to(device)
zerosum_model_2 = zerosum_model_2.to(device)

zerosum_optimizer_1 = optim.SGD(zerosum_model_1.parameters(), lr=0.01, momentum=0.9)
zerosum_optimizer_2 = optim.SGD(zerosum_model_2.parameters(), lr=0.01, momentum=0.9)

def brier_score(outputs, targets, num_classes):
    one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    return torch.mean((one_hot_targets - outputs).pow(2))

def strictly_proper_scoring_rule(outputs, targets, num_classes):
    # The Brier score is a strictly proper scoring rule that measures the accuracy of probabilistic predictions
    # It is the mean squared difference between the predicted probability assigned to the possible outcomes and the actual outcome
    # Here, we assume that 'outputs' are the predicted probabilities of each class (after softmax)
    # and 'targets' are the ground truth labels in one-hot encoded format
    return brier_score(outputs, targets, num_classes=num_classes)

for epoch in range(5):
    zerosum_running_score_1 = 0.0
    zerosum_running_score_2 = 0.0
    for inputs, actions in trainloader:
        # Zero Sum model 1 training
        zerosum_optimizer_1.zero_grad()
        outputs_1 = zerosum_model_1(inputs)
        # Apply softmax to get probabilities
        probabilities_1 = torch.nn.functional.softmax(outputs_1, dim=1)
        loss_1 = strictly_proper_scoring_rule(probabilities_1, actions, actions.max().item() + 1)
        loss_1.backward()
        zerosum_optimizer_1.step()

        # Zero Sum model 2 training
        zerosum_optimizer_2.zero_grad()
        outputs_2 = zerosum_model_2(inputs)
        # Apply softmax to get probabilities
        probabilities_2 = torch.nn.functional.softmax(outputs_2, dim=1)
        loss_2 = strictly_proper_scoring_rule(probabilities_2, actions, actions.max().item() + 1)
        loss_2.backward()
        zerosum_optimizer_2.step()

        # Calculate the zero-sum score for this batch
        batch_score_1 = loss_2.item() - loss_1.item()
        batch_score_2 = loss_1.item() - loss_2.item()
        zerosum_running_score_1 += batch_score_1
        zerosum_running_score_2 += batch_score_2

    print(f'Epoch {epoch+1}, Zero Sum Score Model 1: {zerosum_running_score_1/len(trainloader)}')
    print(f'Epoch {epoch+1}, Zero Sum Score Model 2: {zerosum_running_score_2/len(trainloader)}')
    # Save the model after each epoch
    torch.save(zerosum_model_1.state_dict(), os.path.join('results', 'models', f'zerosum_model_epoch_{epoch+1}.pth'))