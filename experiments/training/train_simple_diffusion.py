import torch
import os
from torch.utils.data import DataLoader
from ..models.simple_diffusion import DiffusionModel
from ..models.simple_unet import UNet
from ..datasets.dummy_dataset import DummyDataset
import argparse
from tqdm import tqdm


def train_with_dummy_dataset(model_path, timesteps, lr, save_dir, num_epochs, batch_size, img_size):
    # Set up the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize diffusion model
    diffusion = DiffusionModel(timesteps=timesteps, device=device)

    # Initialize U-Net model
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize the DummyDataset and DataLoader
    dataset = DummyDataset(img_size=img_size, num_samples=batch_size * 10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    os.makedirs(save_dir, exist_ok=True)
    for epoch in tqdm(range(num_epochs), desc="Epochs", leave=False):
        model.train()
        running_loss = 0.0

        for batch_idx, (x0, _) in enumerate(dataloader):
            x0 = x0.to(device)

            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (x0.size(0),), device=device)

            # Forward diffusion
            noisy_imgs, noise = diffusion.forward_diffusion(x0, t)

            # Reverse diffusion: Predict noise using U-Net
            optimizer.zero_grad()
            pred_noise = model(noisy_imgs, t.unsqueeze(1).float())
            loss = torch.mean((pred_noise - noise) ** 2)  # MSE loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")

    # Save the trained U-Net model
    torch.save(model.state_dict(), os.path.join(save_dir, "dummy_unet_diffusion_model.pth"))
    print("Training complete and model saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='experiments/trained_models/dummy_unet_diffusion_model.pth', help='Path to trained U-Net model.')
    parser.add_argument('--timesteps', type=int, default=500, help='Number of timesteps for diffusion.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training.')
    parser.add_argument('--save_dir', type=str, default='experiments/trained_models/', help='Directory to save trained model.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--img_size', type=int, default=64, help='Image size.')
    args = parser.parse_args()

    train_with_dummy_dataset(args.model_path, args.timesteps, args.lr, args.save_dir, args.num_epochs, args.batch_size, args.img_size)
