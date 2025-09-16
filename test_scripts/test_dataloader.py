# Process starts: Imports.
from torch.utils.data import DataLoader  # Dataloader.
from datasets.modelnet_dataset import ModelNetDataset  # Dataset.
# Process ends.

# Process starts: Test.
def test_dataloader():
    ds = ModelNetDataset(root='ModelNet40', train=True)  # DS.
    loader = DataLoader(ds, batch_size=2)  # Loader.
    for points, labels in loader:
        print(points.shape, labels)  # Print.
        break  # One batch.
# Process ends.

if __name__ == '__main__':
    test_dataloader()  # Run.
