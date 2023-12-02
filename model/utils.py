import torch
import os


def brier_score(outputs, targets, num_classes):
    one_hot_targets = torch.nn.functional.one_hot(
        targets, num_classes=num_classes
    ).float()
    return torch.mean((one_hot_targets - outputs).pow(2))


def strictly_proper_scoring_rule(outputs, targets, num_classes):
    # The Brier score is a strictly proper scoring rule that measures the accuracy of probabilistic predictions
    # It is the mean squared difference between the predicted probability assigned to the possible outcomes and the actual outcome
    # Here, we assume that 'outputs' are the predicted probabilities of each class (after softmax)
    # and 'targets' are the ground truth labels in one-hot encoded format
    return brier_score(outputs, targets, num_classes=num_classes)


# Validation loop
def validate_model(model, valloader, criterion):
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for val_inputs, val_actions in valloader:
            val_inputs, val_actions = val_inputs.to(device), val_actions.to(device)
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_actions).item()
    return val_loss / len(valloader)


# Save model function
def save_model(model, filename):
    path = os.path.join("results", "models", filename)
    torch.save(model.state_dict(), path)


# Determine the device to use
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# Set the random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
