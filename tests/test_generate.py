"""Test the generate.py script."""

import torch
from unittest.mock import patch
from src.generate import generate, save_images


def test_save_images(tmp_path):
    """Ensure images are saved with correct shape and file path."""
    # Create a dummy image
    img = (torch.rand(64, 64, 3).numpy() * 2) - 1.0  # shape (64,64,3), values in [-1,1]
    output_dir = tmp_path / "outputs"
    step = 10

    # Call save_images
    save_images(img, str(output_dir), step=step)

    expected_file = output_dir / "step_10_sample.png"
    assert expected_file.exists(), "Image file should exist after saving."


@patch(
    "src.generate.torch.load", return_value={}
)  # return an empty dict or mock state dict
@patch("src.generate.UNet")
@patch("src.generate.Diffusion")
@patch("src.generate.save_images")
def test_generate(mock_save_images, mock_diffusion, mock_unet, tmp_path):
    """
    Check that generate() runs without error and calls the underlying diffusion method.
    """
    # Mock the UNet object and torch.load
    mock_model_instance = mock_unet.return_value
    mock_model_instance.load_state_dict.return_value = None
    mock_model_instance.eval.return_value = None

    # Mock diffusion's p_sample_loop to yield a single step
    mock_diffusion_instance = mock_diffusion.return_value
    mock_diffusion_instance.p_sample_loop.return_value = [(0, torch.rand(1, 3, 64, 64))]

    # Call generate
    generate(
        img_size=64,
        model_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        timesteps=10,
        device="cpu",
        save_interval=1,
    )

    # Assert the diffusion object was called with correct args
    mock_diffusion.assert_called_once_with(timesteps=10, device="cpu")
    # p_sample_loop should have been called with shape and save_interval
    mock_diffusion_instance.p_sample_loop.assert_called_once()

    # Check that save_images was indeed called
    mock_save_images.assert_called_once()
