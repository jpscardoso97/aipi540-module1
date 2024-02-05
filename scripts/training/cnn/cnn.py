#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('../../scripts')
from data_loader import load_data


# Set the device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load in data from data_loader.py
data_dir = "../../data/raw"
resize_option = False
loaders = load_data(data_dir, customized_size=resize_option)


# In[3]:


print('Number of training samples: {}'.format(len(loaders['train'].sampler)))
print('Number of validation samples: {}'.format(len(loaders['valid'].sampler)))
print('Number of test samples: {}'.format(len(loaders['test'].sampler)))


# In[49]:


import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=38):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 125 * 125, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# In[50]:


# Initialize the model
model = CNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[51]:


total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")


# In[52]:


num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for inputs, labels in loaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loaders['train'])

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.2%}')


# In[ ]:


# Saving model
torch.save(model.state_dict(), 'simple_cnn_model.pth')

