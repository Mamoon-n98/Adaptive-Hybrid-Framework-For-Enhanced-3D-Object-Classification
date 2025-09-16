# Process starts: Imports.
import torch  # PyTorch.
from torch.utils.data import Dataset  # Dataset.
import numpy as np  # Numpy for bin loading.
import os  # Paths.
from datasets.augmentation import augment_points  # Aug.
from datasets.preprocess import preprocess_points  # Preprocess.
# Process ends: Imports.

# Process starts: KittiDataset class.
class KittiDataset(Dataset):
    """
    Loader for KITTI point clouds (.bin).
    """
    def __init__(self, root='KITTI', train=True, num_points=16384):
        super(KittiDataset, self).__init__()  # Init.
        self.root = root  # Root.
        self.train = train  # Train.
        self.num_points = num_points  # Points.
        split = 'training' if train else 'testing'  # Split.
        self.files = [os.path.join(root, split, 'velodyne', f) for f in os.listdir(os.path.join(root, split, 'velodyne')) if f.endswith('.bin')]  # List bin files.
        # Assume labels in label folder, for classification assume scene class or something; for simplicity, random labels for demo.
        self.labels = np.random.randint(0, 10, len(self.files))  # Placeholder labels (adapt for real task).

    def __len__(self):
        return len(self.files)  # Len.

    def __getitem__(self, idx):
        points = np.fromfile(self.files[idx], dtype=np.float32).reshape(-1, 4)  # Load bin (x,y,z,i).
        if points.shape[0] > self.num_points:
            points = points[np.random.choice(points.shape[0], self.num_points, replace=False)]  # Sample.
        label = self.labels[idx]  # Label.
        points = preprocess_points(points)  # Preprocess.
        if self.train:
            points = augment_points(points)  # Aug.
        return torch.from_numpy(points).float(), torch.tensor(label).long()  # Return.
# Process ends: Class.