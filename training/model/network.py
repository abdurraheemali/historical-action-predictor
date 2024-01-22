import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ActionPredictor(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()  # type: ignore
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize models, optimizers, and other components
def initialize_components(
    num_features: int,
    num_classes: int,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
) -> tuple[ActionPredictor, nn.Module, optim.SGD, ReduceLROnPlateau]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionPredictor(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )
    return model, criterion, optimizer, scheduler
