import torch
import numpy as np

class DiffusionModel:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.beta = np.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.tensor(np.cumprod(self.alpha), device=device, dtype=torch.float32)
        self.device = device

    def forward_diffusion(self, x0, t):
        batch_size = x0.size(0)
        noise = torch.randn_like(x0).to(self.device)

        # Index alpha_cumprod for each time step in the batch and reshape for broadcasting
        sqrt_alpha_t = torch.sqrt(self.alpha_cumprod[t]).view(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alpha_cumprod[t]).view(batch_size, 1, 1, 1)

        xt = sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise
        return xt, noise
