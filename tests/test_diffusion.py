"""Test diffusion module."""

import torch
from src.diffusion import Diffusion, cosine_beta_schedule


def test_cosine_beta_schedule():
    timesteps = 10
    betas = cosine_beta_schedule(timesteps=timesteps)
    assert betas.shape == (timesteps,), "Beta schedule should match number of timesteps"
    # Check that betas are within [0, 1)
    assert torch.all(betas >= 0) and torch.all(betas < 1), "Betas must be in [0,1)."


def test_diffusion_init():
    timesteps = 10
    diffusion = Diffusion(timesteps=timesteps, device="cpu")
    assert diffusion.betas.shape == (
        timesteps,
    ), "Diffusion betas should match timesteps."
    assert diffusion.alphas.shape == (
        timesteps,
    ), "Diffusion alphas should match timesteps."
    assert diffusion.alphas_cumprod.shape == (
        timesteps,
    ), "alphas_cumprod should match timesteps."


def test_q_sample():
    diffusion = Diffusion(timesteps=10, device="cpu")
    x_start = torch.randn(2, 3, 64, 64)  # batch of 2 images
    t = torch.tensor([1, 5], dtype=torch.long)  # different timesteps for each sample
    out = diffusion.q_sample(x_start, t)
    assert out.shape == x_start.shape, "q_sample output must be same shape as input."


def test_p_sample_with_mock_model():
    """
    Here we mock a model(x, t) -> predicted_noise.
    We'll check if p_sample returns an image of the right shape.
    """

    class MockModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)  # Predict zero noise for simplicity

    diffusion = Diffusion(timesteps=10, device="cpu")
    x = torch.randn(2, 3, 64, 64)  # batch
    t = 5
    model = MockModel()

    x_prev = diffusion.p_sample(model, x, t)
    assert x_prev.shape == x.shape, "p_sample should return image of same shape."
    # For t>0, random noise is added. Hard to test numerical correctness, but shape check is good.


def test_p_sample_loop_with_mock_model():
    class MockModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    diffusion = Diffusion(timesteps=4, device="cpu")
    model = MockModel()
    shape = (1, 3, 16, 16)
    save_interval = 1
    # Collect all steps into a list to test
    steps_list = list(diffusion.p_sample_loop(model, shape, save_interval))

    # We expect timesteps + 1 yields
    assert len(steps_list) == 4, "Should yield for each t in [3..0], total 4 steps."
    for t, x in steps_list:
        assert x.shape == shape, "Each sample should have the requested shape."
        assert isinstance(t, int), "Timestep should be an integer."
