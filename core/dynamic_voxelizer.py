# Process starts: Imports.
import torch
import spconv.pytorch as spconv
import time
# Process ends: Imports complete.

# Process starts: Helper function to hash points to voxel indices.
def hash_points(points, voxel_size):
    """
    Convert point coordinates to voxel indices.
    :param points: Tensor of shape [batch_size, num_points, 3] or [N, 3].
    :param voxel_size: Float, size of voxel grid.
    :return: Tensor of voxel indices [N, 4] (batch_idx, x, y, z).
    """
    if points.dim() == 3:  # Shape [batch_size, num_points, 3]
        batch_size, num_points, _ = points.shape
        batch_idx = torch.arange(batch_size, device=points.device).view(-1, 1).expand(-1, num_points).reshape(-1, 1)
        points_flat = points.view(-1, 3)  # Flatten to [batch_size * num_points, 3]
    else:  # Shape [N, 3]
        points_flat = points
        batch_idx = torch.zeros(points_flat.shape[0], 1, device=points.device)
    voxel_coords = torch.floor(points_flat / voxel_size).long()
    voxel_coords = torch.cat([batch_idx, voxel_coords], dim=1)  # [N, 4]
    return voxel_coords
# Process ends: Helper function complete.

# Process starts: DynamicVoxelizer class.
class DynamicVoxelizer(torch.nn.Module):
    """
    Convert point clouds to sparse voxel tensors with dynamic voxel sizes.
    """
    def __init__(self, voxel_size=0.05, max_voxels=100000):
        super(DynamicVoxelizer, self).__init__()
        self.voxel_size = voxel_size
        self.max_voxels = max_voxels

    def forward(self, points):
        """
        :param points: Tensor of shape [batch_size, num_points, 3].
        :return: spconv.SparseConvTensor.
        """
        batch_size, num_points, _ = points.shape
        points_flat = points.view(-1, 3)  # [batch_size * num_points, 3]
        voxel_coords = hash_points(points_flat, self.voxel_size)  # [N, 4]
        features = torch.ones((points_flat.shape[0], 1), device=points.device)  # [N, 1]
        spatial_shape = [int(torch.max(voxel_coords[:, i]).item() + 1) for i in range(1, 4)]
        sparse_tensor = spconv.SparseConvTensor(
            features=features,  # [N, 1]
            indices=voxel_coords,  # [N, 4]
            spatial_shape=spatial_shape,  # [X, Y, Z]
            batch_size=batch_size
        )
        if sparse_tensor.indices.shape[0] > self.max_voxels:
            indices = torch.randperm(sparse_tensor.indices.shape[0], device=points.device)[:self.max_voxels]
            sparse_tensor.features = sparse_tensor.features[indices]
            sparse_tensor.indices = sparse_tensor.indices[indices]
        return sparse_tensor
# Process ends: DynamicVoxelizer class defined.
