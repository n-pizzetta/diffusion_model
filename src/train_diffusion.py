"""Train the diffusion model on CelebA dataset."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from src.diffusion import Diffusion
from src.unet import UNet
import os
import yaml
import argparse

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)  # Load configuration file

# Hyperparameters
BATCH_SIZE = config["batch_size"]
EPOCHS = config["epochs"]
LR = config["learning_rate"]
TIMESTEPS = config["timesteps"]
IMAGE_SIZE = config["image_size"]
DEVICE = config["device"]
NUM_WORKERS = config["num_workers"]
MODEL_SAVE_DIR = config["model_save_dir"]


# Custom Dataset
class CelebADataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npy")
        ]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        img = np.load(self.data_files[idx])
        img = torch.from_numpy(img).float()
        return img


def train(
    batch_size,
    img_size,
    epochs,
    learning_rate,
    model_save_dir,
    num_workers,
    timesteps,
    device,
):
    # Prepare data
    dataset = CelebADataset(f"../data/processed/celeba_{img_size}")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # Initialize model and diffusion
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    diffusion = Diffusion(timesteps=timesteps, device=device)
    mse_loss = nn.MSELoss()

    # Training loop
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        print(f"Epoch {epoch+1}/{epochs}")
        pbar = tqdm(dataloader, desc="Batches", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            t = torch.randint(0, timesteps, (batch.size(0),), device=device).long()
            noise = torch.randn_like(batch).to(device)
            x_t = diffusion.q_sample(batch, t, noise)
            pred_noise = model(x_t, t)
            loss = mse_loss(noise, pred_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        # Save model checkpoint
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, "diffusion.pth")
        torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Input batch size for training (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=IMAGE_SIZE,
        help=f"Size of the image (default: {IMAGE_SIZE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of epochs to train (default: {EPOCHS})",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=LR, help=f"Learning rate (default: {LR})"
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default=MODEL_SAVE_DIR,
        help=f"Path to save the model (default: {MODEL_SAVE_DIR})",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of workers for DataLoader (default: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TIMESTEPS,
        help=f"Number of timesteps for diffusion (default: {TIMESTEPS})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device to train the model on (default: {DEVICE})",
    )
    args = parser.parse_args()

    train(
        args.batch_size,
        args.img_size,
        args.epochs,
        args.learning_rate,
        args.model_save_dir,
        args.num_workers,
        args.timesteps,
        args.device,
    )
