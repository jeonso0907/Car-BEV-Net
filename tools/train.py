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

        # Calculate the size after the convolutional and pooling layers
        self._to_linear = None
        self.convs(torch.randn(1, 1, 800, 800))  # Initialize to calculate the size

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 1)

    def convs(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        if self._to_linear is None:
            self._to_linear = x.view(x.size(0), -1).shape[1]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
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

# Check for CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Setup dataset and dataloader
dataset = CarDataset('D:/projects/Car-BEV-Net/data/car_clusters')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Batch size set to 16

# Initialize model, optimizer, and loss function
model = BEVNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# Training loop
num_epochs = 50
best_loss = float('inf')  # Initialize best loss to infinity
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (bev_images, angles) in enumerate(dataloader):
        bev_images, angles = bev_images.to(device), angles.to(device)  # Move data to CPU

        optimizer.zero_grad()
        outputs = model(bev_images)
        loss = loss_function(outputs.squeeze(), angles)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / (i + 1):.4f}')  # Update with running loss

    avg_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    # Save the model if it has the best loss so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'D:/projects/Car-BEV-Net/model/best_model.pth')
        print(f"Saved best model with loss {best_loss:.4f}")

print("Training complete")
