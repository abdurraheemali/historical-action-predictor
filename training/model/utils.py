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
    device = get_device()
    model.eval()
    with torch.no_grad():
        for val_inputs, val_actions in valloader:
            val_inputs, val_actions = val_inputs.to(device), val_actions.to(device)
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_actions).item()
    return val_loss / len(valloader)


def save_model(model, filename):
    dir_path = os.path.join("results", "models")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    path = os.path.join(dir_path, filename)
    torch.save(model.state_dict(), path)


# Determine the device to use
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# Set the random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_ece(outputs, labels, n_bins=10):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = torch.nn.functional.softmax(outputs, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=outputs.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
