# Process starts: Imports.
import torch
from torch.utils.data import DataLoader
import os
import time
from datasets.modelnet_dataset import ModelNetDataset
from configs.modelnet40_config import dataset, model as model_config
from core.dynamic_voxelizer import DynamicVoxelizer
from models.pointnet import PointNet
from models.voxelnet import VoxelNet
from models.fusion import FeatureFusion
from models.bls import BLS
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from configs.default_args import get_default_args
# Process ends: Imports complete.

# Process starts: Model definition.
class HybridModel(nn.Module):
    def __init__(self, num_classes, num_points=2048):
        super(HybridModel, self).__init__()
        self.pointnet = PointNet(num_points=num_points)
        self.voxelnet = VoxelNet(num_classes=num_classes)
        self.fusion = FeatureFusion()
        self.bls = BLS(num_classes=num_classes)

    def forward(self, points, voxels):
        point_feats = self.pointnet(points)
        voxel_feats = self.voxelnet(voxels)
        fused_feats = self.fusion(point_feats, voxel_feats)
        logits = self.bls(fused_feats)
        return logits
# Process ends: Model definition complete.

# Process starts: Get model function.
def get_model(config):
    """
    Initialize the hybrid model based on config.
    :param config: Model configuration dictionary.
    :return: Initialized model.
    """
    return HybridModel(num_classes=config['num_classes'], num_points=config['num_points'])
# Process ends: Get model function complete.

# Process starts: Train one epoch function.
def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Train the model for one epoch.
    :return: Average loss and accuracy.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    voxelizer = DynamicVoxelizer(voxel_size=0.05, max_voxels=100000).to(device)

    for batch_idx, (points, labels) in enumerate(loader):
        points, labels = points.to(device), labels.to(device)
        optimizer.zero_grad()
        voxels = voxelizer(points)
        outputs = model(points, voxels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy
# Process ends: Train one epoch function complete.

# Process starts: Validation function.
def validate(model, loader, device):
    """
    Validate the model.
    :return: Validation accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    voxelizer = DynamicVoxelizer(voxel_size=0.05, max_voxels=100000).to(device)
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            voxels = voxelizer(points)
            outputs = model(points, voxels)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total
# Process ends: Validation function complete.

# Process starts: Main training function.
def train(config):
    """
    Main training loop.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Load datasets.
    train_dataset = ModelNetDataset(root=config['dataset']['root'], train=True, num_points=config['dataset']['num_points'])
    val_dataset = ModelNetDataset(root=config['dataset']['root'], train=False, num_points=config['dataset']['num_points'])
    train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['dataset']['batch_size'], shuffle=False)
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # Initialize model, optimizer, and scheduler.
    model = get_model(config['model']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    # Setup TensorBoard.
    writer = SummaryWriter('outputs/tensorboard')

    # Training loop.
    print("Starting training process...")
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_acc = validate(model, val_loader, device)
        scheduler.step()

        # Log metrics.
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}/{config['training']['epochs']}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save checkpoint.
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'outputs/checkpoints/epoch_{epoch}.pth')

    writer.close()
    print("Training process completed.")
# Process ends: Main training function complete.

# Process starts: Main execution.
if __name__ == '__main__':
    config = get_default_args()  # Load configuration from YAML.
    train(config)  # Run training.
# Process ends: Main execution complete.