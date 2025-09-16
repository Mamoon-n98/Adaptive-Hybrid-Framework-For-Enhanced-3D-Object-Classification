# Process starts: Imports.
import torch  # PyTorch.
import time  # For timing.
# Process ends: Imports.

# Process starts: Hash points function.
def hash_points(points, voxel_size):
    """
    Hash points to voxel coordinates.
    :param points: (N, 3 or 4).
    :param voxel_size: Scalar.
    :return: Voxel coords (N, 3).
    """
    coords = torch.floor(points[:, :3] / voxel_size).long()  # Floor divide.
    return coords  # Return.
# Process ends: Hashing done.

# Process starts: Timing decorator.
def time_function(func):
    """
    Decorator to time function execution.
    """
    def wrapper(*args, **kwargs):
        start = time.time()  # Start time.
        result = func(*args, **kwargs)  # Call function.
        end = time.time()  # End time.
        print(f"{func.__name__} took {end - start:.4f} seconds.")  # Print time.
        return result  # Return result.
    return wrapper  # Return wrapped.
# Process ends: Decorator defined.