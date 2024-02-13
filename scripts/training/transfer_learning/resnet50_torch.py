import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from torch.utils.data import DataLoader

# Change the path to the directory where the scripts are located
sys.path.append('/home/zachxing/AIPI/aipi540-module1')
sys.path.append('/home/zachxing/AIPI/aipi540-module1/scripts')

from data_loader import load_data
from utils.early_stopping import EarlyStopping

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

data_dir = "../../../data/cropped"

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

early_stopping = EarlyStopping(patience=2, restore_best_weights=True)

for epoch in range(80):
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

    # Check if validation metric has stopped improving
    if early_stopping(running_loss, base_model):
        print("Early stopping triggered!")
        early_stopping.restore_best_model(base_model)
        break


# save model to file
torch.save(base_model.state_dict(), '../../../models/transfer_learning.pth')

# Validation
base_model.eval()
correct = 0
total = 0
preds = []
trues = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = base_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        preds.extend(predicted.cpu().numpy())
        trues.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

cm = confusion_matrix(trues, preds)
print('Confusion matrix:')
print(cm)

#save plot to file
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(38)
plt.xticks(tick_marks, range(38), rotation=45)
plt.yticks(tick_marks, range(38))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')

# Print metrics: accuracy, precision, recall and F1 score
accuracy = correct / total
precision = precision_score(trues, preds, average='weighted')
recall = recall_score(trues, preds, average='weighted')
f1 = f1_score(trues, preds, average='weighted')
print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1:.4f}')

# Save metrics to file
with open('metrics1.txt', 'w') as f:
    # write the training data folder
    f.write(f'Training data folder: {data_dir}\n')
    f.write('\n')
    # write the hyperparameters
    f.write('Hyperparameters\n')
    f.write('Batch size: 4\n')
    f.write('Learning rate: 0.001\n')
    f.write('Momentum: 0.9\n')
    f.write('Epochs: 80\n')
    f.write('\n')
    # write the metrics
    f.write('Metrics\n')
    f.write(f'Accuracy: {accuracy:.4f}\n')
    f.write(f'Precision: {precision:.4f}\n')
    f.write(f'Recall: {recall:.4f}\n')
    f.write(f'F1 score: {f1:.4f}\n')


