import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from training.model.network import ActionPredictor
from training.historical_datasets import ProbIdentityDataset


def brier_score(
    outputs: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    one_hot_targets = torch.nn.functional.one_hot(
        targets, num_classes=num_classes
    ).float()
    return torch.mean((one_hot_targets - outputs).pow(2))


def conditional_brier_score(
    outputs: torch.Tensor, targets: torch.Tensor, chosen_actions: torch.Tensor
) -> torch.Tensor:
    chosen_action_probs = torch.gather(outputs, 1, chosen_actions.unsqueeze(1))
    chosen_action_targets = torch.gather(targets, 1, chosen_actions.unsqueeze(1))
    return torch.mean((chosen_action_targets - chosen_action_probs).pow(2))


def strictly_proper_scoring_rule(
    outputs: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    # The Brier score is a strictly proper scoring rule that measures the accuracy of probabilistic predictions
    # It is the mean squared difference between the predicted probability assigned to the possible outcomes and the actual outcome
    # Here, we assume that 'outputs' are the predicted probabilities of each class (after softmax)
    # and 'targets' are the ground truth labels in one-hot encoded format
    return brier_score(outputs, targets, num_classes=num_classes)


def calculate_accuracy(
    model: ActionPredictor, dataloader: DataLoader[ProbIdentityDataset]
) -> float:
    device = get_device()
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            output_probs = torch.nn.functional.softmax(outputs, dim=1)
            loss += torch.nn.functional.cross_entropy(output_probs, labels).item()
    return loss / len(dataloader)


def calculate_chosen_option_accuracy(
    model: ActionPredictor, dataloader: DataLoader[ProbIdentityDataset]
) -> float:
    correct = 0
    total = 0
    device = get_device()
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted_classes = torch.argmax(outputs, dim=1)
            labels = torch.argmax(labels, dim=1)
            total += labels.size(0)
            correct += (predicted_classes == labels).sum().item()
    return correct / total


def calculate_other_options_accuracy(
    model: ActionPredictor, dataloader: DataLoader[ProbIdentityDataset]
) -> float:
    device = get_device()
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            other_outputs = outputs.clone()
            predicted_classes = torch.argmax(outputs, dim=1)
            other_outputs[predicted_classes == 1] = 0
            other_labels = labels.clone()
            other_labels[predicted_classes == 1] = 0
            loss = torch.nn.functional.cross_entropy(other_outputs, other_labels).item()
    return loss / len(dataloader)


# Validation loop
def validate_model(
    model: ActionPredictor,
    valloader: DataLoader[ProbIdentityDataset],
    criterion: nn.CrossEntropyLoss,
) -> float:
    val_loss = 0.0
    device = get_device()
    model.eval()
    with torch.no_grad():
        for val_inputs, val_actions in valloader:
            val_inputs, val_actions = val_inputs.to(device), val_actions.to(device)
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_actions).item()
    return val_loss / len(valloader)


def save_model(model: ActionPredictor, filename: str) -> None:
    dir_path = os.path.join("results", "models")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    path = os.path.join(dir_path, filename)
    torch.save(model.state_dict(), path)  # type: ignore


def load_model(model: ActionPredictor, filename: str) -> ActionPredictor:
    dir_path = os.path.join("results", "models")
    path = os.path.join(dir_path, filename)
    model.load_state_dict(state_dict=torch.load(f=path))  # type: ignore
    model.eval()  # Set the model to evaluation mode
    return model


# Determine the device to use
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# Set the random seed for reproducibility
def set_seed(seed: int) -> None:
    torch.manual_seed(seed=seed)  # type: ignore
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_ece(
    outputs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10
) -> torch.Tensor:
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # softmaxes = torch.nn.functional.softmax(outputs, dim=1)
    # confidences, predictions = torch.max(softmaxes, 1)
    # accuracies = predictions.eq(labels)
    confidences = torch.nn.functional.sigmoid(outputs)
    accuracies = labels

    ece = torch.zeros(1, device=outputs.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
