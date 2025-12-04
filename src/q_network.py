from typing import Sequence

import torch
import torch.nn as nn

from mlp import MLP


class QNetwork(nn.Module):
    """
    Scalar Q(s,a) for continuous MPO.
    Input: [obs, action] concatenated.
    Output: [B] scalar Q-values.
    """

    def __init__(
        self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int] = (256, 256)
    ):
        super().__init__()
        self.backbone = MLP(obs_dim + act_dim, 1, hidden_sizes)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        obs:    [B, obs_dim]
        action: [B, act_dim]
        returns Q: [B]
        """
        x = torch.cat([obs, action], dim=-1)
        q = self.backbone(x)
        return q.squeeze(-1)  # [B]
