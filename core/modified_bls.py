# Process starts: Imports.
import torch  # PyTorch.
import torch.nn as nn  # NN.
# Process ends: Imports.

# Process starts: ModifiedBLS class.
class ModifiedBLS(nn.Module):
    """
    Enhanced Broad Learning System for hybrid inputs, with parallel mapping.
    """
    def __init__(self, in_dim, num_feature_maps=10, num_enhance_nodes=100, out_dim=10):
        super(ModifiedBLS, self).__init__()  # Init.
        self.num_feature_maps = num_feature_maps  # Number of random feature maps.
        self.num_enhance_nodes = num_enhance_nodes  # Enhancement nodes.
        # Random weights for feature mapping (frozen).
        self.feature_weights = nn.Parameter(torch.randn(in_dim, num_feature_maps * in_dim), requires_grad=False)  # Random proj.
        # Enhancement layer.
        self.enhance_weights = nn.Parameter(torch.randn(num_feature_maps * in_dim, num_enhance_nodes), requires_grad=False)  # Random enhance.
        # Output layer (trainable).
        self.output = nn.Linear(num_feature_maps * in_dim + num_enhance_nodes, out_dim)  # Final linear.

    # Process starts: Forward.
    def forward(self, x):
        """
        BLS forward pass.
        :param x: Fused features.
        :return: Output logits.
        """
        # Feature mapping.
        features = torch.matmul(x, self.feature_weights)  # Random projection.
        features = torch.tanh(features)  # Activation.
        # Enhancement nodes.
        enhance = torch.matmul(features, self.enhance_weights)  # Another projection.
        enhance = torch.tanh(enhance)  # Activation.
        # Concat.
        combined = torch.cat([features, enhance], dim=1)  # Combined input.
        # Output.
        out = self.output(combined)  # Linear output.
        return out  # Return.
    # Process ends: BLS computation done.
# Process ends: Class.