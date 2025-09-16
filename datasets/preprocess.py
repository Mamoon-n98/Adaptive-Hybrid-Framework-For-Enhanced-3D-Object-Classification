# Process starts: Imports.
import numpy as np  # Numpy.
from scipy.spatial import KDTree  # For nearest neighbors.
# Process ends: Imports.

# Process starts: Preprocess function.
def preprocess_points(points):
    """
    Denoise, complete point cloud.
    :param points: Numpy array.
    :return: Processed points.
    """
    # Denoise: Remove outliers (simple threshold).
    mean = np.mean(points[:, :3], axis=0)  # Mean.
    dist = np.linalg.norm(points[:, :3] - mean, axis=1)  # Distances.
    points = points[dist < np.std(dist) * 2.5]  # Remove >2.5 std.
    # Completion: Simple upsample if < num_points, but assume fixed.
    return points  # Return.
# Process ends: Preprocess done.