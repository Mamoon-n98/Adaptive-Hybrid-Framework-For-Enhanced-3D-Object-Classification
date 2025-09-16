import torch
from torch.utils.data import Dataset
import os
import numpy as np
import logging
import pickle
import time
import re
from multiprocessing import Pool, cpu_count
from datasets.augmentation import augment_points
from datasets.preprocess import preprocess_points

# Logging setup
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_off(file_path):
    """
    Read a .off file and return vertices as (N,3) numpy array.
    """
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()
                     if line.strip() and not line.startswith('#')]
            if not lines:
                return None

            first_line = lines[0].lower()
            num_vertices = None
            vertex_start = -1

            if first_line == 'off':
                parts = lines[1].split()
                if len(parts) < 3:
                    return None
                num_vertices = int(parts[0])
                vertex_start = 2
            elif re.match(r'^off\d+', first_line):
                match = re.match(r'^off(\d+)\s+(\d+)\s+(\d+)$',
                                 first_line, re.IGNORECASE)
                if not match:
                    return None
                num_vertices = int(match.group(1))
                vertex_start = 1
            else:
                return None

            points = []
            for line in lines[vertex_start:vertex_start + num_vertices]:
                try:
                    coords = list(map(float, line.split()[:3]))
                    if len(coords) == 3:
                        points.append(coords)
                except ValueError:
                    return None

            points = np.array(points, dtype=np.float32)
            return points if points.shape[0] > 0 else None
    except Exception:
        return None


def validate_and_convert(args):
    """
    Worker: validate .off/.npy and convert if needed.
    """
    file_path, class_idx, num_points = args
    npy_path = file_path.replace(".off", ".npy")

    # Case 1: already valid .npy
    if os.path.exists(npy_path):
        try:
            pts = np.load(npy_path)
            if pts.shape[0] > 0:
                return (npy_path, class_idx)
        except Exception:
            pass  # fallthrough to reconvert

    # Case 2: read .off and convert
    pts = read_off(file_path)
    if pts is not None and pts.shape[0] > 0:
        try:
            np.save(npy_path, pts.astype(np.float32))
            return (npy_path, class_idx)
        except Exception as e:
            logger.warning(f"Failed to save {npy_path}: {e}")
            return (file_path, class_idx)  # fallback to raw .off

    # Case 3: unrecoverable file → keep it, will fallback later
    logger.warning(f"Unrecoverable file: {file_path}")
    return (file_path, class_idx)


class ModelNetDataset(Dataset):
    def __init__(self, root, train=True, num_points=2048):
        super(ModelNetDataset, self).__init__()
        self.root = root
        self.train = train
        self.num_points = num_points
        self.split = 'train' if train else 'test'

        cache_file = os.path.join(root, f"{self.split}_valid_files.pkl")
        start_time = time.time()

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.files, self.classes, self.class_to_idx = pickle.load(f)
        else:
            self.classes = sorted([
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d)) and not d.startswith('.')
            ])
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            candidate_files = []

            for cls in self.classes:
                split_dir = os.path.join(root, cls, self.split)
                if not os.path.isdir(split_dir):
                    logger.warning(f"Missing {split_dir}, skipping {cls}")
                    continue
                for fn in os.listdir(split_dir):
                    if fn.endswith('.off'):
                        candidate_files.append(
                            (os.path.join(split_dir, fn),
                             self.class_to_idx[cls],
                             self.num_points)
                        )

            num_workers = min(cpu_count(), 16)
            with Pool(num_workers) as p:
                validated = p.map(validate_and_convert, candidate_files)

            self.files = [item for item in validated if item is not None]

            with open(cache_file, 'wb') as f:
                pickle.dump((self.files, self.classes, self.class_to_idx), f)

        logger.info(f"Found {len(self.files)} {self.split} samples "
                    f"in {time.time() - start_time:.2f}s")

        if not self.files:
            raise ValueError(f"No valid files in {root}/{self.split}")

    def __len__(self):
        return len(self.files)

    def _fix_num_points(self, points):
        n = points.shape[0]
        if n == 0:
            return np.random.normal(0, 0.1,
                                    (self.num_points, 3)).astype(np.float32)
        if n > self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
            points = points[idx]
        elif n < self.num_points:
            pad = self.num_points - n
            idx = np.random.choice(n, pad, replace=True)
            points = np.concatenate([points, points[idx]], axis=0)
        return points

    def __getitem__(self, idx):
        file_path, label = self.files[idx]

        # Load npy if available
        points = None
        if file_path.endswith(".npy"):
            try:
                points = np.load(file_path)
            except Exception:
                points = None

        # Otherwise read .off
        if points is None or points.shape[0] == 0:
            if file_path.endswith(".off"):
                points = read_off(file_path)
            if points is None or points.shape[0] == 0:
                logger.warning(f"Invalid/empty: {file_path}, using random")
                points = np.random.normal(0, 0.1,
                                          (self.num_points, 3)).astype(np.float32)

        # Ensure correct point count
        points = self._fix_num_points(points)

        # Preprocess + augment
        points = preprocess_points(points)
        if self.train:
            points = augment_points(points)
        points = self._fix_num_points(points)

        return torch.from_numpy(points).float(), torch.tensor(label).long()
