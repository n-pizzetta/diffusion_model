import torch
from diffusion import Diffusion
from unet import UNet
import matplotlib.pyplot as plt
import os
import yaml
import argparse

with open('../configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f) # Load configuration file

TIMESTEPS = config['timesteps']
IMAGE_SIZE = config['image_size']
DEVICE = config['device']
MODEL_DIR = config['model_save_dir']
OUTPUT_DIR = config['output_dir']

def save_images(img, output_dir, step):
    os.makedirs(output_dir, exist_ok=True)
    # img is in [-1,1], scale to [0,255]
    img = ((img + 1) * 127.5).clip(0, 255).astype('uint8')
    plt.imsave(os.path.join(output_dir, f'step_{step}_sample.png'), img)



def generate(img_size, model_dir, output_dir, timesteps, device, save_interval):
    model = UNet().to(device)
    model_path = os.path.join(model_dir, 'diffusion.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    diffusion = Diffusion(timesteps=timesteps, device=device)
    shape = (1, 3, img_size, img_size)

    with torch.no_grad():
        # Get images for each step in the denoising process
        for step, img in diffusion.p_sample_loop(model, shape, save_interval=save_interval):
            # img is a torch tensor at this step
            img_np = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            save_images(img_np, output_dir, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=IMAGE_SIZE, help='Size of the image')
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR, help='Directory containing the model')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory to save generated images')
    parser.add_argument('--timesteps', type=int, default=TIMESTEPS, help='Number of timesteps')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device (cpu or cuda)')
    parser.add_argument('--save_interval', type=int, default=50, help='Interval to save images')
    args = parser.parse_args()

    generate(args.img_size, args.model_dir, args.output_dir, args.timesteps, args.device, args.save_interval)
