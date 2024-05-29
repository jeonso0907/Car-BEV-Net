import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle

# Define BEVNet model
class BEVNet(nn.Module):
    def __init__(self):
        super(BEVNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 40 * 40, 512)  # Adjust according to input size
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 40 * 40)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom Dataset for loading BEV images and angles
class CarDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(os.path.join(self.data_dir, self.files[idx]), 'rb') as f:
            data = pickle.load(f)
        bev_image = data['bev_image'][np.newaxis, :, :]  # Add channel dimension
        angle = data['angle']
        return torch.tensor(bev_image, dtype=torch.float32), torch.tensor(angle, dtype=torch.float32)

# Setup dataset and dataloader
dataset = CarDataset('car_bev_data')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, optimizer, and loss function
model = BEVNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for bev_images, angles in dataloader:
        optimizer.zero_grad()
        outputs = model(bev_images)
        loss = loss_function(outputs.squeeze(), angles)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
