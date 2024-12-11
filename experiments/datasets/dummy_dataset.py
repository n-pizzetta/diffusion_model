import torch
from torch.utils.data import Dataset
import numpy as np

class DummyDataset(Dataset):
    """Generates random data for testing U-Net and diffusion models."""
    def __init__(self, img_size=64, num_samples=100):
        self.img_size = img_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random image and a "label" (e.g., blurred version or noisy version)
        img = np.random.rand(self.img_size, self.img_size).astype(np.float32)
        label = (img > 0.5).astype(np.float32)  # Example: binary segmentation
        img = torch.tensor(img).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(label).unsqueeze(0)
        return img, label
