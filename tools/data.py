import os
import numpy as np
import open3d as o3d
import pickle
from tqdm import tqdm  # Import tqdm for the progress bar

# Paths to KITTI dataset
pcd_path = "D:/projects/Car-BEV-Net/data/kitti_sample/velodyne"
label_path = "D:/projects/Car-BEV-Net/data/kitti_sample/label_2"

# Function to load PCD or BIN file
def load_pcd(file_path):
    # KITTI data is usually stored in .bin format for LiDAR
    pcd = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    pcd = pcd[:, :3]  # discard the reflectance value, only x, y, z needed
    return pcd

# Function to load labels
def load_labels(file_path):
    with open(file_path, 'r') as f:
        labels = f.readlines()
    return labels

# Extract car points from point cloud
def extract_car_points(points, labels):
    car_points = []
    car_angles = []
    for label in labels:
        parts = label.split()
        if parts[0] == 'Car':
            x, y, z, l, w, h, ry = float(parts[11]), float(parts[12]), float(parts[13]), float(parts[8]), float(
                parts[9]), float(parts[10]), float(parts[14])
            mask = ((points[:, 0] > x - l / 2) & (points[:, 0] < x + l / 2) &
                    (points[:, 1] > y - w / 2) & (points[:, 1] < y + w / 2) &
                    (points[:, 2] > z - h / 2) & (points[:, 2] < z + h / 2))
            car_points.append(points[mask])
            car_angles.append(ry)
    return car_points, car_angles

# Convert point cloud to BEV image
def point_cloud_to_bev(points, res=0.1, z_max=2.5, z_min=-2.5):
    bev_image = np.zeros((int(80 / res), int(80 / res)), dtype=np.float32)
    points = points[(points[:, 2] > z_min) & (points[:, 2] < z_max)]
    for x, y, z in points:
        x_img = int((x + 40) / res)
        y_img = int((y + 40) / res)
        if 0 <= x_img < bev_image.shape[0] and 0 <= y_img < bev_image.shape[1]:
            bev_image[x_img, y_img] = max(bev_image[x_img, y_img], z)
    bev_image = (bev_image - z_min) / (z_max - z_min)
    return bev_image

# Save BEV images and angles
def save_bev_images(car_points, car_angles, base_filename, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, (points, angle) in enumerate(zip(car_points, car_angles)):
        bev_image = point_cloud_to_bev(points)
        data = {'bev_image': bev_image, 'angle': angle}
        output_file = os.path.join(output_dir, f'{base_filename}_car_{i}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

# Main extraction process
output_dir = "D:/projects/Car-BEV-Net/data/car_clusters"
files = os.listdir(pcd_path)
for file in tqdm(files, desc="Processing files"):  # Use tqdm here to show progress
    pcd_file = os.path.join(pcd_path, file)
    label_file = os.path.join(label_path, file.replace('.bin', '.txt'))
    base_filename = os.path.splitext(file)[0]  # Extract the base name without the extension

    points = load_pcd(pcd_file)
    labels = load_labels(label_file)

    car_points, car_angles = extract_car_points(points, labels)
    save_bev_images(car_points, car_angles, base_filename, output_dir)