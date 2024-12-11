import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ..datasets.dummy_dataset import DummyDataset
from ..models.simple_unet import UNet
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def train_dummy_unet(img_size, num_samples, num_epochs, batch_size, lr, save_dir):

    # Initialize dataset and dataloader
    dataset = DummyDataset(img_size=img_size, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize U-Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Epochs", leave=False):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(dataloader, desc="Batches", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}")

    # Save model for evaluation
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}dummy_unet.pth")
    print("Dummy U-Net trained and saved!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=64, help='Images size.')
    parser.add_argument('--num_samples', type=int, default=3*64, help='Number of samples to generate.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--save_dir', type=str, default='experiments/trained_models/', help='Directory to save trained model.')
    args = parser.parse_args()

    train_dummy_unet(args.img_size, args.num_samples, args.num_epochs, args.batch_size, args.lr, args.save_dir)
