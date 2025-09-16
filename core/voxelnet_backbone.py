# Process starts: Imports for VoxelNet backbone.
import torch  # PyTorch.
import torch.nn as nn  # NN.
import spconv.pytorch as spconv  # Sparse conv.
# Process ends: Imports.

# Process starts: VoxelNetBackbone class.
class VoxelNetBackbone(nn.Module):
    """
    3D CNN backbone on sparse voxels using spconv.
    """
    def __init__(self, in_channels=4, out_channels=128):
        super(VoxelNetBackbone, self).__init__()  # Super init.
        # Sparse conv layers.
        self.conv1 = spconv.SparseConv3d(in_channels, 64, kernel_size=3, stride=1, padding=1)  # First conv.
        self.bn1 = spconv.SparseBatchNorm(64)  # BN.
        self.relu1 = spconv.SparseReLU()  # ReLU.
        self.conv2 = spconv.SparseConv3d(64, out_channels, kernel_size=3, stride=2, padding=1)  # Downsample conv.
        self.bn2 = spconv.SparseBatchNorm(out_channels)  # BN.
        self.relu2 = spconv.SparseReLU()  # ReLU.

    # Process starts: Forward.
    def forward(self, sparse_tensor):
        """
        Process sparse voxels.
        :param sparse_tensor: Input from voxelizer.
        :return: Voxel features.
        """
        x = self.conv1(sparse_tensor)  # Conv1.
        x = self.bn1(x)  # BN1.
        x = self.relu1(x)  # ReLU1.
        x = self.conv2(x)  # Conv2.
        x = self.bn2(x)  # BN2.
        x = self.relu2(x)  # ReLU2.
        # Dense features for fusion.
        features = x.dense()  # Convert to dense tensor.
        return features.mean([2,3,4])  # Global mean pool (B, C).
    # Process ends: Backbone processing done.
# Process ends: Class.