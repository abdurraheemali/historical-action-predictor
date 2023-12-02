import torch
import torch.nn as nn
import torch.optim as optim
from datasets import HistoricalDataset
from torch.utils.data import  TensorDataset, DataLoader

class ActionPredictor(nn.Module):
    def __init__(self):
        super(ActionPredictor, self).__init__()

        self.fc1 = nn.Linear(784, 128)

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
# Load the data from the .npy files
features = np.load('data/generated_features.npy')
labels = np.load('data/generated_labels.npy')

# Convert the numpy arrays to torch tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Create a TensorDataset from the tensors
historical_dataset = TensorDataset(features_tensor, labels_tensor)

# Create a DataLoader from the TensorDataset
trainloader = DataLoader(historical_dataset, batch_size=64, shuffle=True)

# Performative Predictor
performative_model = ActionPredictor()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(performative_model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(5):
    running_loss = 0.0
    for inputs, actions in trainloader:
        optimizer.zero_grad()
        outputs = performative_model(inputs)  # Corrected model variable name
        # Create a one-hot encoded tensor of the actions
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=5).float()
        # Multiply the outputs with the one-hot encoded actions
        performative_outputs = outputs * actions_one_hot
        loss = criterion(performative_outputs, actions)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

# Zero Sum Predictor
zerosum_model = ActionPredictor()
optimizer = optim.SGD(zerosum_model.parameters(), lr=0.01, momentum=0.9)

# Zero Sum Predictor (with zero-sum normalization)
zerosum_model_1 = ActionPredictor()
zerosum_model_2 = ActionPredictor()
zerosum_optimizer_1 = optim.SGD(zerosum_model_1.parameters(), lr=0.01, momentum=0.9)
zerosum_optimizer_2 = optim.SGD(zerosum_model_2.parameters(), lr=0.01, momentum=0.9)

def strictly_proper_scoring_rule(outputs, targets):
    # The Brier score is a strictly proper scoring rule that measures the accuracy of probabilistic predictions
    # It is the mean squared difference between the predicted probability assigned to the possible outcomes and the actual outcome
    # Here, we assume that 'outputs' are the predicted probabilities of each class (after softmax)
    # and 'targets' are the ground truth labels in one-hot encoded format
    one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=outputs.size(1)).float()
    return torch.mean((one_hot_targets - outputs) ** 2)

for epoch in range(5):
    zerosum_running_loss_1 = 0.0
    zerosum_running_loss_2 = 0.0
    for inputs, actions in trainloader:
        # Zero Sum model 1 training
        zerosum_optimizer_1.zero_grad()
        outputs_1 = zerosum_model_1(inputs)
        # Apply softmax to get probabilities
        probabilities_1 = torch.nn.functional.softmax(outputs_1, dim=1)
        loss_1 = strictly_proper_scoring_rule(probabilities_1, actions)
        loss_1.backward()
        zerosum_optimizer_1.step()
        zerosum_running_loss_1 += loss_1.item()

        # Zero Sum model 2 training
        zerosum_optimizer_2.zero_grad()
        outputs_2 = zerosum_model_2(inputs)
        # Apply softmax to get probabilities
        probabilities_2 = torch.nn.functional.softmax(outputs_2, dim=1)
        loss_2 = strictly_proper_scoring_rule(probabilities_2, actions)
        loss_2.backward()
        zerosum_optimizer_2.step()
        zerosum_running_loss_2 += loss_2.item()

        # Normalize the outputs for zero-sum competition
        with torch.no_grad():
            outputs_1 -= outputs_1.mean(dim=1, keepdim=True)
            outputs_2 -= outputs_2.mean(dim=1, keepdim=True)

    # Calculate the zero-sum scores for each predictor
    zerosum_score_1 = zerosum_running_loss_1 - zerosum_running_loss_2
    zerosum_score_2 = zerosum_running_loss_2 - zerosum_running_loss_1

    print(f'Epoch {epoch+1}, Zero Sum Score Model 1: {zerosum_score_1/len(trainloader)}')
    print(f'Epoch {epoch+1}, Zero Sum Score Model 2: {zerosum_score_2/len(trainloader)}')