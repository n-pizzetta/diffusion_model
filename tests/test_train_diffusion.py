"""Test the training pipeline for the diffusion model."""

import torch
from unittest.mock import patch
from src.train_diffusion import train


@patch("src.train_diffusion.CelebADataset")
@patch("src.train_diffusion.DataLoader")
@patch("src.train_diffusion.os.makedirs")
@patch("torch.save")
def test_train_diffusion(
    mock_torch_save, mock_os_makedirs, mock_dataloader, mock_celeba_dataset
):
    """
    Mocks the entire training pipeline so no real file I/O or heavy GPU usage occurs.
    """
    # Mock the dataset length
    mock_celeba_dataset.return_value.__len__.return_value = 4
    # Mock each batch from the dataloader
    batch = torch.randn(2, 3, 64, 64)
    mock_dataloader.return_value = [(batch)]

    # Run the train function
    train(
        batch_size=2,
        img_size=64,
        epochs=1,
        learning_rate=1e-4,
        model_save_dir="fake_dir",
        num_workers=0,
        timesteps=10,
        device="cpu",
    )

    # Check if torch.save was called (meaning we presumably saved a checkpoint)
    mock_torch_save.assert_called_once()
    # Check if directories are created
    mock_os_makedirs.assert_called_once_with("fake_dir", exist_ok=True)
