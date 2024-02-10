import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from torch.utils.data import DataLoader

sys.path.append('../../')
from data_loader import load_data

print("PyTorch version:", torch.__version__)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device available:', device)

# Data augmentation and normalization for training
train_transforms = transforms.Compose([
    transforms.RandomRotation(40),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Just normalization for validation
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = "../../../data/raw"

# Dataloaders
loaders = load_data(data_dir, customized_size=False)
train_loader = loaders['train']
val_loader = loaders['valid']

# Load the pre-trained model, without the top layer
base_model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
for param in base_model.parameters():
    param.requires_grad = False

# Replace the top layer for fine-tuning
num_ftrs = base_model.fc.in_features
base_model.fc = nn.Linear(num_ftrs, 38)

base_model = base_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(base_model.parameters(), lr=0.001, momentum=0.9)

# Training loop
base_model.train()
for epoch in range(10):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = base_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# save model to file
torch.save(base_model.state_dict(), '../../models/transfer_learning.pth')

# Validation
base_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = base_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Validation accuracy:', correct / total)
