# Process starts: Imports.
import torch  # PyTorch.
from core.dynamic_voxelizer import DynamicVoxelizer  # Voxelizer.
# Process ends.

# Process starts: Test function.
def test_voxelizer():
    voxelizer = DynamicVoxelizer()  # Instance.
    points = torch.rand(1000, 3)  # Random points.
    voxels = voxelizer(points)  # Voxelize.
    print(voxels)  # Print.
# Process ends.

if __name__ == '__main__':
    test_voxelizer()  # Run.