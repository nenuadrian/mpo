import math
import torch
import torch.nn as nn
from mpo.utils import compute_weights_and_temperature_loss_torch


class MPOLoss(nn.Module):
    """
    Small helper that encapsulates MPO dual variables:
      - log_temperature (scalar)
      - log_alpha_mean (scalar)
      - log_alpha_stddev (scalar)

    Exposes:
      - temperature(): softplus(log_temperature) + eps
      - alphas(): (alpha_mean, alpha_std) = softplus of the logs
      - compute_weights_and_temperature_loss(q_values, epsilon): wrapper around utils helper
    """

    def __init__(
        self,
        eta: float = 1.0,
        init_log_alpha_mean: float = 0.0,
        init_log_alpha_stddev: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        init_log_temperature = math.log(max(eta, 1e-8))
        self.log_temperature = nn.Parameter(
            torch.tensor([init_log_temperature], dtype=torch.float32, device=device)
        )
        self.log_alpha_mean = nn.Parameter(
            torch.tensor([init_log_alpha_mean], dtype=torch.float32, device=device)
        )
        self.log_alpha_stddev = nn.Parameter(
            torch.tensor([init_log_alpha_stddev], dtype=torch.float32, device=device)
        )
        self._eps = 1e-8

    def temperature(self) -> torch.Tensor:
        # primal temperature (softplus ensures positivity)
        return torch.nn.functional.softplus(self.log_temperature) + self._eps

    def alphas(self) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_mean = torch.nn.functional.softplus(self.log_alpha_mean) + self._eps
        alpha_std = torch.nn.functional.softplus(self.log_alpha_stddev) + self._eps
        return alpha_mean, alpha_std

    def compute_weights_and_temperature_loss(
        self, q_values: torch.Tensor, epsilon: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q_values: [K, B] (same convention as utils helper)
        returns (normalized_weights [K,B], temperature_loss scalar tensor)
        """
        return compute_weights_and_temperature_loss_torch(
            q_values=q_values, epsilon=float(epsilon), temperature=self.temperature()
        )
