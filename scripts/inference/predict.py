import os
import sys
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from PIL import Image
import io

from scripts.data_loader import get_padding_transform

current_directory = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(current_directory, '../../models/transfer_learning-100epoch_withCropped.pth')

class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.model.eval()
        self.transform = get_padding_transform(customized_size=False, target_size=(224, 224))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self, model_path):
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 38)
        
        # Load the model weights
        with open(model_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        
        model.load_state_dict(torch.load(buffer, map_location=self.device))
        model.to(self.device)
        
        return model

    def predict(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            
        return torch.nn.functional.softmax(output, dim=1).cpu().numpy().tolist()[0]

