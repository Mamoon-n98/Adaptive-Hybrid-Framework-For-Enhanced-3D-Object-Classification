# Process starts: Imports.
import torch  # PyTorch.
import numpy as np  # Numpy.
# Process ends: Imports.

# Process starts: Augment function.
def augment_points(points):
    """
    Inject real-world noise, jitter, etc.
    :param points: Numpy or tensor.
    :return: Augmented points.
    """
    if isinstance(points, torch.Tensor):
        points = points.numpy()  # To numpy.
    # Rotation.
    theta = np.random.uniform(0, 2*np.pi)  # Random angle.
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0,0,1]])  # Rot Z.
    points[:, :3] = points[:, :3] @ rot_matrix  # Apply.
    # Jitter.
    jitter = np.random.normal(0, 0.01, points.shape)  # Noise.
    points += jitter  # Add.
    # Scale.
    scale = np.random.uniform(0.8, 1.2)  # Random scale.
    points[:, :3] *= scale  # Apply.
    return points  # Return.
# Process ends: Augmentation done.
