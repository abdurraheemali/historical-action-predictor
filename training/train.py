import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from historical_datasets import (
    HistoricalDatasetConfig,
    ProbIdentityDataset,
)
from typing import Literal
import os
import logging
import json
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.network import initialize_components, ActionPredictor
from model.utils import (
    set_seed,
    get_device,
    calculate_ece,
    conditional_brier_score,
    validate_model,
    save_model,
    load_model,
    calculate_accuracy,
    calculate_chosen_option_accuracy,
    calculate_other_options_accuracy,
)

# Load configuration
with open(os.path.join(os.path.dirname(__file__), "config.json")) as config_file:
    config = json.load(config_file)

SEED = config["SEED"]
NUM_FEATURES = config["NUM_FEATURES"]
NUM_CLASSES = config["NUM_CLASSES"]
LEARNING_RATE = config["LEARNING_RATE"]
MOMENTUM = config["MOMENTUM"]
NUM_EPOCHS = config["NUM_EPOCHS"]
BATCH_SIZE = config["BATCH_SIZE"]
VALIDATION_SPLIT = config["VALIDATION_SPLIT"]
NUM_EPISODES = config["NUM_EPISODES"]
EPISODE_LENGTH = config["EPISODE_LENGTH"]
MODEL_DIR = config["MODEL_DIR"]

# Configure logging
logging_config = config["LOGGING"]

filename: str = os.path.join(*logging_config["FILENAME"].split("/"))

if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))
logging.basicConfig(
    filename=filename,
    filemode=logging_config["FILEMODE"],
    level=logging.getLevelName(logging_config["LEVEL"]),
    format=logging_config["FORMAT"],
)

set_seed(SEED)
device: Literal["cuda", "cpu"] = get_device()


def train_performative_model(
    model: ActionPredictor,
    trainloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    valloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    num_epochs: int = NUM_EPOCHS,
) -> None:
    best_loss = float("inf")

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for inputs, actions in trainloader:
            inputs, actions = inputs.to(device), actions.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            probabilities = torch.nn.functional.sigmoid(outputs)
            chosen_actions = torch.argmax(probabilities, dim=1)

            conditional_brier_score(probabilities, actions, chosen_actions)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = validate_model(model=model, valloader=valloader, criterion=criterion)
        logging.info(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

        if val_loss < best_loss:
            best_loss = val_loss

            save_model(model, "best_performative_model.pth")

        scheduler.step(val_loss)
        save_model(model, f"performative_model_epoch_{epoch+1}.pth")


# Define the Zero Sum Predictor training function
def train_zero_sum_models(
    model_1: ActionPredictor,
    model_2: ActionPredictor,
    trainloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer_1: optim.Optimizer,
    optimizer_2: optim.Optimizer,
    num_epochs: int = NUM_EPOCHS,
) -> None:
    for epoch in tqdm(range(num_epochs)):
        zerosum_running_score_1 = 0.0
        zerosum_running_score_2 = 0.0
        for inputs, actions in trainloader:
            inputs, actions = inputs.to(device), actions.to(device)

            # Zero the gradients for both optimizers
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            outputs_1 = model_1(inputs)

            probabilities_1 = torch.nn.functional.sigmoid(outputs_1)
            outputs_2 = model_2(inputs)

            probabilities_2 = torch.nn.functional.sigmoid(outputs_2)

            max_probabilities = torch.max(probabilities_1, probabilities_2)
            chosen_actions = torch.argmax(max_probabilities, dim=1)

            score_1 = conditional_brier_score(probabilities_1, actions, chosen_actions)
            score_2 = conditional_brier_score(probabilities_2, actions, chosen_actions)

            loss_1 = score_1 - score_2.detach()
            loss_2 = score_2 - score_1.detach()

            loss_1.backward(retain_graph=True)  # type: ignore
            optimizer_1.step()

            optimizer_2.zero_grad()

            loss_2.backward()  # type: ignore
            optimizer_2.step()

            # Calculate the zero-sum score for this batch
            batch_score_1 = loss_1.item() - loss_2.item()
            batch_score_2 = loss_2.item() - loss_1.item()
            zerosum_running_score_1 += batch_score_1
            zerosum_running_score_2 += batch_score_2

        logging.info(
            f"Epoch {epoch+1}, Zero Sum Score Model 1: {zerosum_running_score_1/len(trainloader)}"
        )
        logging.info(
            f"Epoch {epoch+1}, Zero Sum Score Model 2: {zerosum_running_score_2/len(trainloader)}"
        )
        # Save the models after each epoch
        save_model(model=model_1, filename=f"zerosum_model_1_epoch_{epoch+1}.pth")
        save_model(model=model_2, filename=f"zerosum_model_2_epoch_{epoch+1}.pth")


def main():
    # Initialize performative model with the number of input features and classes
    performative_model, criterion, optimizer, scheduler = initialize_components(
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
    )

    # Load and prepare data
    config = HistoricalDatasetConfig(
        num_episodes=NUM_EPISODES,
        num_classes=NUM_CLASSES,
        episode_length=EPISODE_LENGTH,
        num_features=NUM_FEATURES,
        transform=None,
    )

    full_dataset = ProbIdentityDataset(config=config)
    validation_split = 0.2
    num_train = int((1 - validation_split) * len(full_dataset))
    num_val = len(full_dataset) - num_train
    train_dataset, val_dataset = random_split(
        dataset=full_dataset, lengths=[num_train, num_val]
    )
    trainloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    train_performative = False
    if train_performative:
        train_performative_model(
            model=performative_model,
            trainloader=trainloader,
            valloader=valloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=NUM_EPOCHS,
        )
    else:
        # Load the performative model instead of training
        performative_model = load_model(
            ActionPredictor(NUM_FEATURES, NUM_CLASSES), "best_performative_model.pth"
        )
    performative_model = performative_model.to(device)

    train_zerosum = True
    if train_zerosum:
        zerosum_model_1 = ActionPredictor(NUM_FEATURES, NUM_CLASSES)
        zerosum_model_2: ActionPredictor = copy.deepcopy(zerosum_model_1)
        train_zero_sum_models(
            model_1=zerosum_model_1,
            model_2=zerosum_model_2,
            trainloader=trainloader,
            optimizer_1=optimizer,
            optimizer_2=optimizer,
            num_epochs=NUM_EPOCHS,
        )

    # Load trained Zero Sum Predictors
    zerosum_model_1 = load_model(
        ActionPredictor(NUM_FEATURES, NUM_CLASSES), "zerosum_model_1_epoch_100.pth"
    )
    zerosum_model_2 = load_model(
        ActionPredictor(NUM_FEATURES, NUM_CLASSES), "zerosum_model_2_epoch_100.pth"
    )
    zerosum_model_1 = zerosum_model_1.to(device)
    zerosum_model_2 = zerosum_model_2.to(device)

    # Calculate ECE for a batch from the validation set
    inputs, labels = next(iter(valloader))
    inputs, labels = inputs.to(device), labels.to(device)
    ece_performative = calculate_ece(performative_model(inputs), labels)
    ece_zerosum_1 = calculate_ece(zerosum_model_1(inputs), labels)
    ece_zerosum_2 = calculate_ece(zerosum_model_2(inputs), labels)

    # Calculate accuracies for each model
    accuracy_performative = calculate_accuracy(performative_model, valloader)
    accuracy_zerosum_1 = calculate_accuracy(zerosum_model_1, valloader)
    accuracy_zerosum_2 = calculate_accuracy(zerosum_model_2, valloader)

    chosen_option_accuracy_performative = calculate_chosen_option_accuracy(
        performative_model, valloader
    )
    chosen_option_accuracy_zerosum_1 = calculate_chosen_option_accuracy(
        zerosum_model_1, valloader
    )
    chosen_option_accuracy_zerosum_2 = calculate_chosen_option_accuracy(
        zerosum_model_2, valloader
    )

    other_options_accuracy_performative = calculate_other_options_accuracy(
        performative_model, valloader
    )
    other_options_accuracy_zerosum_1 = calculate_other_options_accuracy(
        zerosum_model_1, valloader
    )
    other_options_accuracy_zerosum_2 = calculate_other_options_accuracy(
        zerosum_model_2, valloader
    )

    # Plot ECE values
    plt.figure(figsize=(10, 6))  # type: ignore
    plt.bar(  # type: ignore
        ["Performative Model", "Zero Sum Model 1", "Zero Sum Model 2"],
        [ece_performative.item(), ece_zerosum_1.item(), ece_zerosum_2.item()],
    )  # type: ignore
    plt.title("Expected Calibration Error for Each Model")  # type: ignore
    plt.ylabel("ECE")  # type: ignore
    plt.savefig("results/ece_plot.png")  # type: ignore

    # Plot average prediction accuracy across models

    plt.figure(figsize=(10, 6))  # type: ignore
    plt.bar(  # type: ignore
        ["Performative Model", "Zero Sum Model 1", "Zero Sum Model 2"],
        [accuracy_performative, accuracy_zerosum_1, accuracy_zerosum_2],
    )
    plt.title("Average Prediction Accuracy for Each Model")  # type: ignore
    plt.ylabel("Accuracy")  # type: ignore
    plt.savefig("results/average_accuracy_plot.png")  # type: ignore

    # Plot accuracy on the chosen option across models
    plt.figure(figsize=(10, 6))  # type: ignore
    plt.bar(  # type: ignore
        ["Performative Model", "Zero Sum Model 1", "Zero Sum Model 2"],
        [
            chosen_option_accuracy_performative,
            chosen_option_accuracy_zerosum_1,
            chosen_option_accuracy_zerosum_2,
        ],
    )
    plt.title("Accuracy on Chosen Option for Each Model")  # type: ignore
    plt.ylabel("Accuracy")  # type: ignore
    plt.savefig("results/chosen_option_accuracy_plot.png")  # type: ignore

    # Plot accuracy on all other options across models
    plt.figure(figsize=(10, 6))  # type: ignore
    plt.bar(  # type: ignore
        x=["Performative Model", "Zero Sum Model 1", "Zero Sum Model 2"],
        height=[
            other_options_accuracy_performative,
            other_options_accuracy_zerosum_1,
            other_options_accuracy_zerosum_2,
        ],
    )
    plt.title(label="Accuracy on Other Options for Each Model")  # type: ignore
    plt.ylabel(ylabel="Accuracy")  # type: ignore
    plt.savefig("results/other_options_accuracy_plot.png")  # type: ignore


if __name__ == "__main__":
    main()
