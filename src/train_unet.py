import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from unet import UNet
import os
import numpy as np
import yaml
from tqdm import tqdm
import argparse

with open('../configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f) # Load configuration file

# Hyperparameters
BATCH_SIZE = config['batch_size']
EPOCHS = config['epochs']
LR = config['learning_rate']
IMAGE_SIZE = config['image_size']
DEVICE = config['device']
NUM_WORKERS = config['num_workers']
MODEL_SAVE_DIR = config['model_save_dir']


# CelebA Dataset Loader
class CelebADataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        img = np.load(self.data_files[idx])
        img = torch.from_numpy(img).float()
        return img
    
def train_unet(batch_size, epochs, img_size, learning_rate, model_save_dir, num_workers, device):

    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dataset and DataLoader
    dataset = CelebADataset(f"../data/processed/celeba_{img_size}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Training
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        model.train()
        running_loss = 0.0
        for imgs in tqdm(dataloader, desc="Batches", leave=False):
            imgs = imgs.to(device)

            optimizer.zero_grad()
            t = torch.tensor(0).to(device)
            outputs = model(imgs, t) # t=0
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # Save the model
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, "unet.pth")
    torch.save(model.state_dict(), model_save_path)
    print("Model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=IMAGE_SIZE, help='Size of the image')
    parser.add_argument('--learning_rate', type=float, default=LR, help='Learning rate')
    parser.add_argument('--model_save_dir', type=str, default=MODEL_SAVE_DIR, help='Directory to save the model')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='Number of workers for DataLoader')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device (cpu or cuda)')
    args = parser.parse_args()

    train_unet(args.batch_size, args.epochs, args.img_size, args.learning_rate, args.model_save_dir, args.num_workers, args.device)
