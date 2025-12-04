from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from mlp import MLP


class GaussianPolicy(nn.Module):
    """
    Tanh-squashed Gaussian policy for continuous Box action spaces.
    a = tanh(z) * action_scale + action_bias
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        super().__init__()
        self.backbone = MLP(obs_dim, 2 * act_dim)
        self.log_std_min = -5.0
        self.log_std_max = 2.0

        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0

        self.register_buffer(
            "action_scale", torch.as_tensor(action_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.as_tensor(action_bias, dtype=torch.float32)
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns mean and log_std for the *pre-tanh* Gaussian.
        obs: [B, obs_dim]
        """
        out = self.backbone(obs)
        mu, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def _sample_pre_tanh(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample pre-tanh Gaussian z, and compute tanh(z) actions and log-probs.
        """
        mu, log_std = self(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        # rsample for reparameterization
        z = dist.rsample()
        a_tanh = torch.tanh(z)
        # Squashed to env bounds
        action = a_tanh * self.action_scale + self.action_bias

        # log pi(a|s): base log_prob minus log|det(d tanh)|
        log_prob = dist.log_prob(z)  # [B, act_dim]
        log_prob = log_prob - torch.log(1.0 - a_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)  # [B]
        return action, log_prob, a_tanh, z

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and return (action, log_prob).
        obs: [B, obs_dim]
        """
        action, log_prob, _, _ = self._sample_pre_tanh(obs)
        return action, log_prob

    def log_prob_pre_tanh(self, obs, z):
        mu, log_std = self(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        return dist.log_prob(z).sum(-1)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log π(a|s) for given (s,a) pair.
        We invert the squashing transform: a_tanh = (a - bias) / scale, z = atanh(a_tanh).
        obs: [B, obs_dim]
        action: [B, act_dim]
        """
        # Map from env action back to [-1,1]
        a_tanh = (action - self.action_bias) / self.action_scale
        a_tanh = torch.clamp(a_tanh, -0.999999, 0.999999)

        # atanh(x) = 0.5 * (log(1+x) - log(1-x))
        z = 0.5 * (torch.log1p(a_tanh) - torch.log1p(-a_tanh))

        mu, log_std = self(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        base_log_prob = dist.log_prob(z)  # [B, act_dim]

        # log |det d(tanh)/dz|^{-1} = - sum log(1 - tanh(z)^2)
        log_det = torch.log(1.0 - a_tanh.pow(2) + 1e-6)
        log_prob = base_log_prob.sum(dim=-1) - log_det.sum(dim=-1)
        return log_prob
