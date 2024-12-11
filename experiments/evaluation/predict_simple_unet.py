import os 
import torch 
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from ..datasets.dummy_dataset import DummyDataset
from ..models.simple_unet import UNet

def evaluate_dummy_unet(model_path, img_size, num_samples, batch_size, save_dir):
    """Evaluate the trained dummy U-Net and save sample outputs."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    os.makedirs(save_dir, exist_ok=True)


    # Generate random samples
    dataset = DummyDataset(img_size=img_size, num_samples=num_samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        for idx, (img, label) in tqdm(enumerate(dataloader), desc="Samples", leave=False):
            
            img, label = img.to(device), label.to(device)
            output = model(img)
            output = torch.sigmoid(output)  # Convert logits to probabilities
            
            # Convert tensors to numpy arrays for visualization
            img_np = img[0].cpu().squeeze().numpy()
            label_np = label[0].cpu().squeeze().numpy()
            output_np = output[0].cpu().squeeze().numpy()

            # Plot and save the results
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img_np, cmap="gray")
            axs[0].set_title("Input")
            axs[1].imshow(label_np, cmap="gray")
            axs[1].set_title("Ground Truth")
            axs[2].imshow(output_np, cmap="gray")
            axs[2].set_title("Model Output")
            plt.tight_layout()
            
            sample_path = os.path.join(save_dir, f"dummy_unet_sample_{idx}.png")
            plt.savefig(sample_path)
            plt.close()
            print(f"Saved sample output to {sample_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='experiments/trained_models/dummy_unet.pth', help='Path to trained model.')
    parser.add_argument('--img_size', type=int, default=64, help='Images size.')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to generate for prediction.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for prediction.')
    parser.add_argument('--save_dir', type=str, default='experiments/results/samples/', help='Directory to save sample prediction outputs.')
    args = parser.parse_args()

    # Evaluate the model and save sample outputs
    evaluate_dummy_unet(args.model_path, args.img_size, args.num_samples, args.batch_size, args.save_dir)