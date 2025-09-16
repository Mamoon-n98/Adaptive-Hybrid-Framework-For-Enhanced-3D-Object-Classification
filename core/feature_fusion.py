# Process starts: Imports.
import torch  # PyTorch.
import torch.nn as nn  # NN.
# Process ends: Imports.

# Process starts: FeatureFusion class.
class FeatureFusion(nn.Module):
    """
    Attention-based multi-scale fusion of point and voxel features.
    """
    def __init__(self, point_dim, voxel_dim, out_dim=256):
        super(FeatureFusion, self).__init__()  # Init.
        # Attention layers.
        self.attn = nn.MultiheadAttention(point_dim + voxel_dim, num_heads=8)  # Multi-head attention.
        self.fc = nn.Linear(point_dim + voxel_dim, out_dim)  # Final linear.

    # Process starts: Forward.
    def forward(self, point_feats, voxel_feats):
        """
        Fuse features.
        :param point_feats: From PointNet.
        :param voxel_feats: From VoxelNet.
        :return: Fused features.
        """
        # Concatenate along channel dim.
        combined = torch.cat([point_feats, voxel_feats], dim=1)  # (B, point_dim + voxel_dim).
        # Attention (reshape for seq len=1).
        combined = combined.unsqueeze(0)  # (1, B, D).
        attn_out, _ = self.attn(combined, combined, combined)  # Self-attention.
        attn_out = attn_out.squeeze(0)  # (B, D).
        # Final projection.
        fused = self.fc(attn_out)  # (B, out_dim).
        return fused  # Return.
    # Process ends: Fusion complete.
# Process ends: Class.