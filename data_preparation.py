import os
import numpy as np
from pypcd import pypcd
import pickle

# Paths to KITTI dataset
pcd_path = "path_to_kitti/velodyne"
label_path = "path_to_kitti/label_2"

# Function to load PCD file
def load_pcd(file_path):
    pc = pypcd.PointCloud.from_path(file_path)
    points = np.vstack((pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z'])).T
    return points

# Function to load labels
def load_labels(file_path):
    with open(file_path, 'r') as f:
        labels = f.readlines()
    return labels

def extract_car_points(points, labels):
    car_points = []
    car_angles = []
    for label in labels:
        parts = label.split()
        if parts[0] == 'Car':
            x, y, z, l, w, h, ry = float(parts[11]), float(parts[12]), float(parts[13]), float(parts[8]), float(parts[9]), float(parts[10]), float(parts[14])
            mask = ((points[:, 0] > x - l / 2) & (points[:, 0] < x + l / 2) &
                    (points[:, 1] > y - w / 2) & (points[:, 1] < y + w / 2) &
                    (points[:, 2] > z - h / 2) & (points[:, 2] < z + h / 2))
            car_points.append(points[mask])
            car_angles.append(ry)
    return car_points, car_angles

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


def save_bev_images(car_points, car_angles, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, (points, angle) in enumerate(zip(car_points, car_angles)):
        bev_image = point_cloud_to_bev(points)
        data = {'bev_image': bev_image, 'angle': angle}
        with open(os.path.join(output_dir, f'car_{i}.pkl'), 'wb') as f:
            pickle.dump(data, f)


# Example usage
output_dir = "car_bev_data"
for file in os.listdir(pcd_path):
    pcd_file = os.path.join(pcd_path, file)
    label_file = os.path.join(label_path, file.replace('.bin', '.txt'))

    points = load_pcd(pcd_file)
    labels = load_labels(label_file)

    car_points, car_angles = extract_car_points(points, labels)
    save_bev_images(car_points, car_angles, output_dir)