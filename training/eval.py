# Process starts: Modify sys.path to include project root for imports.
import sys  # Import sys for path manipulation.
import os  # Import os for directory operations.
import time  # Import time for timestamp logging.
# Get the absolute path to the project root (assuming notebook is in hybrid_model_project/ or a subdirectory).
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))  # Go up one directory from current working dir.
sys.path.insert(0, project_root)  # Add project root to path.
# Process ends: Path setup complete.

# Process starts: Imports.
import torch  # PyTorch for tensor operations.
from training.train import evaluate, DynamicVoxelizer, HierarchicalPointNet, VoxelNetBackbone, FeatureFusion, ModifiedBLS, ModelNetDataset, KittiDataset, DataLoader, get_default_args, nn  # Import from train.py.
# Process ends: Imports complete.

# Process starts: Main evaluation function.
def main():
    config = get_default_args()  # Load configuration from YAML.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU or CPU.
    print(f"Evaluating on device: {device}")  # Log device.
    # Initialize test dataset based on config.
    if 'modelnet' in config['dataset']['root'].lower():
        ds = ModelNetDataset(config['dataset']['root'], train=False, num_points=config['dataset']['num_points'])  # Load ModelNet test dataset.
    else:
        ds = KittiDataset(config['dataset']['root'], train=False, num_points=config['dataset']['num_points'])  # Load KITTI test dataset.
    loader = DataLoader(ds, batch_size=config['dataset']['batch_size'])  # Create test DataLoader.
    print(f"Loaded {len(ds)} test samples.")  # Log dataset size.
    # Initialize model components.
    voxelizer = DynamicVoxelizer().to(device)  # Dynamic voxelizer.
    pointnet = HierarchicalPointNet(out_channels=config['model']['point_out_dim']).to(device)  # Hierarchical PointNet.
    voxelnet = VoxelNetBackbone(out_channels=config['model']['voxel_out_dim']).to(device)  # VoxelNet backbone.
    fusion = FeatureFusion(config['model']['point_out_dim'], config['model']['voxel_out_dim'], config['model']['fusion_out_dim']).to(device)  # Feature fusion module.
    bls = ModifiedBLS(config['model']['fusion_out_dim'], out_dim=config['model']['num_classes']).to(device)  # Modified BLS.
    # Load saved model weights.
    if not os.path.exists('checkpoints/best_model.pth'):
        raise FileNotFoundError("Model weights not found at 'checkpoints/best_model.pth'. Run training first.")
    bls.load_state_dict(torch.load('checkpoints/best_model.pth'))  # Load saved model weights.
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for evaluation.
    print("Starting evaluation process...")
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Record start time.
    acc = evaluate(pointnet, voxelnet, fusion, bls, voxelizer, loader, device, criterion)  # Run evaluation.
    print(f"[{start_time}] Test Accuracy: {acc:.4f}")  # Print test accuracy with timestamp.
    if acc > 0.93:
        print("Achieved target test accuracy > 93%.")
    else:
        print("Test accuracy below 93%. Consider further training or hyperparameter tuning.")
    # Process ends: Evaluation complete.
# Process ends: Main evaluation function complete.

# Process starts: Main execution.
if __name__ == '__main__':
    main()  # Run evaluation.
# Process ends: Main execution complete.
