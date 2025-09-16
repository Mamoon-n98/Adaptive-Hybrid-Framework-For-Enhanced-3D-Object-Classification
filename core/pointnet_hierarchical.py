# Process starts: Importing libraries for hierarchical PointNet.
import torch  # Import PyTorch.
import torch.nn as nn  # Import NN modules.
import torch.nn.functional as F  # Import functional operations.
from torch_cluster import fps  # Import farthest point sampling from torch_cluster.
# Process ends: Libraries imported.

# Process starts: Defining HierarchicalPointNet class.
class HierarchicalPointNet(nn.Module):
    """
    Hierarchical PointNet using FPS and MLPs for multi-scale feature extraction.
    """
    def __init__(self, in_channels=3, out_channels=1024, num_levels=3):
        # Initialize parent class.
        super(HierarchicalPointNet, self).__init__()
        self.num_levels = num_levels  # Number of hierarchical levels.
        # Define MLPs for each level.
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, 64, 1),  # 1D conv for point features.
                nn.BatchNorm1d(64),  # Batch normalization.
                nn.ReLU(),  # Activation.
                nn.Conv1d(64, 128, 1),  # Next conv.
                nn.BatchNorm1d(128),  # BN.
                nn.ReLU()  # ReLU.
            ) for _ in range(num_levels)
        ])
        # Global max pool for final features.
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # Adaptive max pooling.

    # Process starts: Forward pass.
    def forward(self, points):
        """
        Extract hierarchical features.
        :param points: (B, N, C) point cloud.
        :return: Hierarchical features.
        """
        features = []  # List to hold features from each level.
        current_points = points.transpose(1, 2)  # Transpose to (B, C, N) for conv1d.
        for level in range(self.num_levels):
            # Apply MLP.
            feat = self.mlps[level](current_points)  # MLP features.
            # Max pool per point.
            feat = feat.max(2)[0]  # (B, 128).
            features.append(feat)  # Append to list.
            # FPS to sample points for next level.
            sample_idx = fps(points[:, :, :3].reshape(-1, 3), ratio=0.5)  # Sample half points.
            current_points = current_points[:, :, sample_idx]  # Downsample.
        # Concatenate all level features.
        hierarchical_feat = torch.cat(features, dim=1)  # (B, sum(out_channels)).
        return hierarchical_feat  # Return features.
    # Process ends: Features extracted.
# Process ends: Class defined.