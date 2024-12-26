"""Preprocess CelebA dataset."""

import os
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import argparse
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)  # Load configuration file

# Hyperparameters
IMAGE_SIZE = config["image_size"]
DICT = {"train": 0, "val": 1, "test": 2}


def preprocess_celeba(input_dir, output_dir, csv_path, n_images, data_type, img_size):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            ),  # Normalize to [-1, 1]
        ]
    )

    os.makedirs(output_dir, exist_ok=True)

    # Read attribute file
    df = pd.read_csv(csv_path)
    df = df[df["partition"] == DICT[data_type]]
    df = df.sample(n=n_images) if n_images is not None else df

    # Preprocess images
    for img_file in tqdm(df["image_id"]):
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path).convert("RGB")  # Convert to RGB
        img_tensor = transform(img)
        img_array = img_tensor.numpy()

        # Save as numpy file
        output_path = os.path.join(output_dir, img_file.replace(".jpg", ".npy"))
        np.save(output_path, img_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_images", type=int, default=1000, help="Number of images to preprocess"
    )
    parser.add_argument(
        "--data_type", type=str, default="train", help="Data type (train, val or test)"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/list_eval_partition.csv",
        help="Path to CelebA partitioning file",
    )
    parser.add_argument(
        "--img_size", type=int, default=IMAGE_SIZE, help="Size of the image"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/celeba",
        help="Directory containing raw images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"data/processed/celeba_{IMAGE_SIZE}",
        help="Directory to save processed images",
    )
    args = parser.parse_args()
    preprocess_celeba(
        args.input_dir,
        args.output_dir,
        args.csv_path,
        args.n_images,
        args.data_type,
        args.img_size,
    )
