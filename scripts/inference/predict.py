import torch
import torchvision
from torchvision import models, transforms
from PIL import Image

# Load the pre-trained model
model = torch.load('../../models/transfer_learning.pth')
model.eval()

# Load and preprocess the input image
input_image = Image.open('input.jpg')
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# Make predictions
with torch.no_grad():
    output = model(input_batch)

# Get the predicted class
_, predicted = torch.max(output, 1)

print('Predicted class:', predicted.item())
