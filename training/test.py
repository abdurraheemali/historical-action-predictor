import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ActionPredictor(nn.Module):
    def __init__(self):
        super(ActionPredictor, self).__init__()

        self.fc1 = nn.Linear(784, 128)

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        # Define the forward pass
        pass
        
    historical_dataset = HistoricalDataset()
    trainloader = torch.utils.data.DataLoader(historical_dataset, batch_size=64, shuffle=True)

    model = ActionPredictor()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(5):
        running_loss = 0.0
        for inputs, actions in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')