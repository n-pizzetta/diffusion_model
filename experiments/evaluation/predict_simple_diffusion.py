import os
import torch
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from ..models.simple_diffusion import DiffusionModel
from ..models.simple_unet import UNet
from ..datasets.dummy_dataset import DummyDataset

def evaluate_reverse_diffusion(model_path, img_size, num_samples, timesteps, device, save_dir):
    """Evaluate and visualize the reverse diffusion process."""
    os.makedirs(save_dir, exist_ok=True)

    # Load the trained model
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Initialize the dataset and create noisy images for evaluation
    dataset = DummyDataset(img_size=img_size, num_samples=num_samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)  # One image at a time

    # Initialize random timesteps and noise scheduler
    diffusion = DiffusionModel(timesteps=timesteps, device=device)

    with torch.no_grad():
        for idx, (x0, _) in tqdm(enumerate(dataloader), desc="Samples", leave=False):
            x0 = x0.to(device)

            # Random timestep for this sample
            t = torch.randint(0, diffusion.timesteps, (x0.size(0),), device=device)

            # Forward diffusion to generate noisy images
            noisy_imgs, noise = diffusion.forward_diffusion(x0, t)

            # Predict noise using the trained U-Net model
            pred_noise = model(noisy_imgs, t.unsqueeze(1).float())
            recovered_imgs = noisy_imgs - pred_noise  # Approximate denoising (not full reverse diffusion)

            # Prepare data for visualization
            noisy_img_np = noisy_imgs[0].cpu().squeeze().numpy()
            recovered_img_np = recovered_imgs[0].cpu().squeeze().numpy()
            original_img_np = x0[0].cpu().squeeze().numpy()

            # Create a figure with three images side by side
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(noisy_img_np, cmap="gray")
            axs[0].set_title("Noisy Image (Forward Diffusion)")
            axs[0].axis("off")

            axs[1].imshow(recovered_img_np, cmap="gray")
            axs[1].set_title("Recovered Image (Approx. Denoising)")
            axs[1].axis("off")

            axs[2].imshow(original_img_np, cmap="gray")
            axs[2].set_title(f"Original Image at t={t.item()} (Ground Truth)")
            axs[2].axis("off")

            plt.tight_layout()

            # Save the combined plot
            save_path_combined = os.path.join(save_dir, f"dummy_diffusion_sample_{idx}.png")
            plt.savefig(save_path_combined)
            plt.close()
            print(f"Saved comparison plot to {save_path_combined}")

            if idx >= num_samples - 1:  # Stop after evaluating the desired number of samples
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='experiments/trained_models/dummy_unet_diffusion_model.pth', help='Path to trained model.')
    parser.add_argument('--img_size', type=int, default=64, help='Image size.')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to evaluate.')
    parser.add_argument('--timesteps', type=int, default=500, help='Number of timesteps for diffusion.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use.')
    parser.add_argument('--save_dir', type=str, default='experiments/results/samples/', help='Directory to save sample outputs.')
    args = parser.parse_args()

    evaluate_reverse_diffusion(args.model_path, args.img_size, args.num_samples, args.timesteps, args.device, args.save_dir)
