import torch
from torch.cuda.amp import autocast
from diffusers import UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import numpy as np

def main(save_dir, timesteps, img_size):
    os.makedirs(save_dir, exist_ok=True)

    # Device setup
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the pre-trained model
    print("Loading pre-trained U-Net model...")
    unet = UNet2DModel.from_pretrained(f"google/ddpm-celebahq-{img_size}").to(device)

    # Load the scheduler
    print("Loading scheduler...")
    scheduler = DDPMScheduler.from_pretrained(f"google/ddpm-celebahq-{img_size}")

    def to_image_array(tensor):
        # Rescale the output to [0, 1]
        img = (tensor.clamp(-1, 1) + 1) / 2
        img = img.cpu().squeeze().permute(1, 2, 0).numpy()
        return img

    import numpy as np

    def generate_image(unet, scheduler, device, timesteps, img_size, num_visualization_images=5):
        # Initialize a random noise tensor (starting point for generation)
        noise = torch.randn(1, 3, img_size, img_size).to(device)

        # Set the scheduler timesteps
        scheduler.set_timesteps(timesteps)
        total_steps = len(scheduler.timesteps)

        # Define which timesteps to visualize
        # Always include the first (most noisy) and last (fully denoised).
        # Distribute the remaining (num_visualization_images - 2) evenly in between.
        if num_visualization_images > total_steps:
            num_visualization_images = total_steps  # Can't have more images than steps

        # If we have only a few steps, just take them all
        if num_visualization_images <= 2:
            # Just first and last if only 2 requested
            indices = [0, total_steps-1] if num_visualization_images == 2 else [0]
        else:
            # We have at least 3 images: first, last, and intermediates
            # Generate intermediate indices spaced between 1 and total_steps-2
            intermediate_indices = np.linspace(1, total_steps-2, num_visualization_images-2, dtype=int)
            indices = [0] + list(intermediate_indices) + [total_steps-1]

        # Convert these indices to actual timestep values
        visualization_steps = [int(scheduler.timesteps[i].item()) for i in indices]

        snapshots = {}

        # Iteratively denoise the image
        for t in tqdm(scheduler.timesteps, desc="Denoising", leave=False):
            with torch.no_grad(), autocast():
                # Predict the noise residual with the U-Net model
                noise_pred = unet(noise, t).sample

            # Update the image (denoising step)
            noise = scheduler.step(noise_pred, t, noise).prev_sample

            # If this timestep is one we want to visualize, store the image
            t_int = int(t.item())
            if t_int in visualization_steps:
                snapshots[t_int] = to_image_array(noise)

        # Ensure the final image at t=0 is recorded if it's not already
        if 0 not in snapshots:
            snapshots[0] = to_image_array(noise)

        return noise, snapshots


    print("Generating image and collecting intermediate snapshots...")
    final_noise, snapshots = generate_image(unet, scheduler, device, timesteps, img_size)

    # Sort the snapshots by timestep in descending order (from noisy to clean)
    sorted_steps = sorted(snapshots.keys(), reverse=True)
    print(f"Collected snapshots for timesteps: {sorted_steps}")

    # Plot the chosen steps side-by-side
    fig, axes = plt.subplots(1, len(sorted_steps), figsize=(4*len(sorted_steps), 4))
    if len(sorted_steps) == 1:
        axes = [axes]  # Make it iterable even if only one image
    for ax, step in zip(axes, sorted_steps):
        ax.imshow(snapshots[step])
        ax.set_title(f"Timestep {step}")
        ax.axis("off")
    plt.tight_layout()

    combined_image_path = os.path.join(save_dir, f"steps_comparison.png")
    plt.savefig(combined_image_path)
    plt.close()

    print(f"Saved combined steps image to {combined_image_path}")

    # Also save just the final generated image (t=0)
    final_image = (final_noise.clamp(-1, 1) + 1) / 2
    final_image = final_image.cpu().squeeze().permute(1, 2, 0).numpy()
    plt.figure(figsize=(4,4))
    plt.imshow(final_image)
    plt.axis("off")
    plt.tight_layout()
    final_image_path = os.path.join(save_dir, f"final_generated_image.png")
    plt.savefig(final_image_path)
    plt.close()
    print(f"Saved final generated image to {final_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='results/generated_images', help='Directory to save generated images.')
    parser.add_argument('--timesteps', type=int, default=50, help='Number of timesteps for denoising.')
    parser.add_argument('--img_size', type=int, default=256, help='Image size for generation.')
    args = parser.parse_args()

    main(args.save_dir, args.timesteps, args.img_size)
