import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from historical_datasets import HistoricalDatasetConfig, HistoricalDataset
import os
import random
import logging
import json
from model.network import initialize_components, ActionPredictor
from model.utils import (
    set_seed,
    get_device,
    brier_score,
    strictly_proper_scoring_rule,
    validate_model,
    save_model,
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
EARLY_STOPPING_PATIENCE = config["EARLY_STOPPING_PATIENCE"]
MODEL_DIR = config["MODEL_DIR"]

# Configure logging
logging_config = config["LOGGING"]
filename = os.path.join(*logging_config["FILENAME"].split("/"))

if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))
logging.basicConfig(
    filename=filename,
    filemode=logging_config["FILEMODE"],
    level=logging.getLevelName(logging_config["LEVEL"]),
    format=logging_config["FORMAT"],
)

set_seed(SEED)
device = get_device()


def train_performative_model(
    model,
    trainloader,
    valloader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=NUM_EPOCHS,
):
    best_loss = float("inf")
    early_stopping_patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, actions in trainloader:
            inputs, actions = inputs.to(device), actions.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = validate_model(model, valloader, criterion)
        logging.info(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            save_model(model, "best_performative_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info("Early stopping triggered")
                break

        scheduler.step(val_loss)
        save_model(model, f"performative_model_epoch_{epoch+1}.pth")


# Define the Zero Sum Predictor training function
def train_zero_sum_models(
    model_1, model_2, trainloader, optimizer_1, optimizer_2, num_epochs=NUM_EPOCHS
):
    for epoch in range(num_epochs):
        zerosum_running_score_1 = 0.0
        zerosum_running_score_2 = 0.0
        for inputs, actions in trainloader:
            inputs, actions = inputs.to(device), actions.to(device)

            # Zero the gradients for both optimizers
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            # Forward pass for both models
            outputs_1 = model_1(inputs)
            probabilities_1 = torch.nn.functional.softmax(outputs_1, dim=1)
            outputs_2 = model_2(inputs)
            probabilities_2 = torch.nn.functional.softmax(outputs_2, dim=1)

            # Compute scores
            score_1 = strictly_proper_scoring_rule(
                probabilities_1, actions, actions.max().item() + 1
            )
            score_2 = strictly_proper_scoring_rule(
                probabilities_2, actions, actions.max().item() + 1
            )

            # Compute the losses as the negation of the other's score
            loss_1 = score_1 - score_2.detach()
            loss_2 = score_2 - score_1.detach()

            loss_1.backward(retain_graph=True)
            optimizer_1.step()

            optimizer_2.zero_grad()

            loss_2.backward()
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
        save_model(model_1, f"zerosum_model_1_epoch_{epoch+1}.pth")
        save_model(model_2, f"zerosum_model_2_epoch_{epoch+1}.pth")


def main():
    # Initialize performative model with the number of input features and classes
    performative_model, criterion, optimizer, scheduler = initialize_components(
        ActionPredictor, NUM_FEATURES, NUM_CLASSES, LEARNING_RATE, MOMENTUM
    )

    # Load and prepare data
    config = HistoricalDatasetConfig(
        num_episodes=NUM_EPISODES,
        num_classes=NUM_CLASSES,
        episode_length=EPISODE_LENGTH,
        num_features=NUM_FEATURES,
        transform=None,
    )

    full_dataset = HistoricalDataset(config=config)
    validation_split = 0.2
    num_train = int((1 - validation_split) * len(full_dataset))
    num_val = len(full_dataset) - num_train
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Train the performative model
    train_performative_model(
        performative_model, trainloader, valloader, criterion, optimizer, scheduler
    )

    # Initialize and train Zero Sum Predictors
    zerosum_model_1 = ActionPredictor(NUM_FEATURES, NUM_CLASSES).to(device)
    zerosum_model_2 = ActionPredictor(NUM_FEATURES, NUM_CLASSES).to(device)
    zerosum_optimizer_1 = optim.SGD(zerosum_model_1.parameters(), lr=0.01, momentum=0.9)
    zerosum_optimizer_2 = optim.SGD(zerosum_model_2.parameters(), lr=0.01, momentum=0.9)
    train_zero_sum_models(
        zerosum_model_1,
        zerosum_model_2,
        trainloader,
        zerosum_optimizer_1,
        zerosum_optimizer_2,
    )


if __name__ == "__main__":
    main()
