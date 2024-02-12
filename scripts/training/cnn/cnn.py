# %%
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


# Set the device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load in data from data_loader.py
data_dir = "../../../data/raw"
# resize_option = False
# loaders = load_data(data_dir, customized_size=resize_option)

# Transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load the dataset
dataset = ImageFolder(data_dir, transform=train_transform)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# %%
print('Number of training samples: {}'.format(len(train_loader.sampler)))
print('Number of validation samples: {}'.format(len(val_loader.sampler)))
# print('Number of test samples: {}'.format(len(loaders['test'].sampler)))

# %%
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=38):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128 * 56 * 56, 38)

    def forward(self, x):
        # Convolutional layers
        x = nn.ReLU()(self.conv1(x))
        x = self.pool1(x)
        x = nn.ReLU()(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dropout
        x = self.dropout(x)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x


# %%
# Initialize the model
model = CNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")

# %%
num_epochs = 20

logging.basicConfig(filename='cnn.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.2%}')

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.2%}')

# %%
# Saving model
torch.save(model.state_dict(), 'simple_cnn_model.pth')


