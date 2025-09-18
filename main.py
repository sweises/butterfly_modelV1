import torch
import torch.nn as nn
import torch.optim as optim

from models.resnet50 import create_model
from utils.dataset import get_dataloaders
from utils.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, classes = get_dataloaders("data")
model = create_model(len(classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model = train_model(model, train_loader, val_loader, device, criterion, optimizer, epochs=10)

torch.save(model.state_dict(), "resnet50_butterflies.pth")
