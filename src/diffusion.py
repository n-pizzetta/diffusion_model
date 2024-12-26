"""Diffusion model for image generation."""

import torch
import torch.nn.functional as F
import numpy as np


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in Nichol & Dhariwal 2021.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999)


class Diffusion:
    def __init__(self, timesteps=1000, device="cuda"):
        self.device = device
        self.timesteps = timesteps
        self.betas = cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (adding noise) at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, model, x, t):
        """
        Sample from the model at timestep t
        """
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]) ** 0.5
        sqrt_recip_alphas_t = (1.0 / self.alphas[t]) ** 0.5

        # Predict noise
        eps_theta = model(x, torch.tensor([t], device=self.device))
        # Compute x_{t-1}
        x_prev = sqrt_recip_alphas_t * (
            x - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_theta
        )

        if t > 0:
            noise = torch.randn_like(x).to(self.device)
            x_prev += noise * betas_t**0.5
        else:
            x_prev = x_prev
        return x_prev

    def p_sample_loop(self, model, shape, save_interval):
        x = torch.randn(shape).to(self.device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, t)
            if t % save_interval == 0 or t == 0:
                yield t, x
