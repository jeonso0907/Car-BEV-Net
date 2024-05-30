import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# Define BEVNet model
class BEVNet(nn.Module):
    def __init__(self):
        super(BEVNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate the size after the convolutional and pooling layers
        self._to_linear = None
        self._initialize_linear_size()

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 1)

    def _initialize_linear_size(self):
        with torch.no_grad():
            sample_input = torch.randn(1, 1, 800, 800)
            x = self.pool(F.relu(self.conv1(sample_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            self._to_linear = x.view(x.size(0), -1).shape[1]
            print(f'Calculated _to_linear size: {self._to_linear}')  # Debug print

    def convs(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self.convs(x)
        print(f'Shape before flattening: {x.shape}')  # Debug print
        x = x.view(x.size(0), -1)
        print(f'Shape after flattening: {x.shape}')  # Debug print
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the saved model weights
device = torch.device("cpu")
model = BEVNet().to(device)
model.load_state_dict(torch.load('D:/projects/Car-BEV-Net/model/best_model.pth'))
model.eval()  # Set the model to evaluation mode

# Function to load a single data sample for inference
def load_sample(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    bev_image = data['bev_image'][np.newaxis, :, :]  # Add channel dimension
    return torch.tensor(bev_image, dtype=torch.float32)

# Example inference with a sample file
sample_file = 'D:/projects/Car-BEV-Net/data/car_clusters/000001_car_0.pkl'  # Replace with your file path
bev_image = load_sample(sample_file).to(device)

# Perform inference
with torch.no_grad():
    prediction = model(bev_image)
    print(f'Predicted angle: {prediction.item()}')

# This function can be used to load any sample for inference
