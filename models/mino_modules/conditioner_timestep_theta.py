import torch
from torch import nn
from .conditioner_timestep import get_sincos_1d_from_grid


class ConditionerTimestepTheta(nn.Module):
    """Joint conditioner on diffusion timestep t and SPDE parameter theta. 
    Produces c = e_t + e_theta of shape (batch, dim*4)
    Drop-in replacement for ConditionerTimestep
    """

    def __init__(self, dim, theta_dim):
        super().__init__()
        cond_dim = dim * 4
        self.dim = dim
        self.cond_dim = cond_dim        # read by encoder/decoder
        self.theta_dim = theta_dim

        # Timestep branch: identical to original ConditionerTimestep
        self.timestep_mlp = nn.Sequential(
            nn.Linear(dim, cond_dim),
            nn.SiLU(),
        )

        # Theta branch (paper Section 3.5): "two linear layers with SiLU
        # activations and layer normalization".
        self.theta_mlp = nn.Sequential(
            nn.Linear(theta_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.LayerNorm(cond_dim),
        )

    def forward(self, timestep, theta):
        # timestep: (batch,) or scalar broadcastable to (batch,)
        # theta:    (batch, theta_dim)
        assert timestep.numel() == len(timestep)
        timestep = timestep.flatten().double()
        t_embed = self.timestep_mlp(
            get_sincos_1d_from_grid(timestep, dim=self.dim)
        )
        # Cast theta to match t_embed dtype (float after MLP projection)
        theta_embed = self.theta_mlp(theta.to(t_embed.dtype))

        return t_embed + theta_embed    # c = e_t + e_theta (Eq. 8)