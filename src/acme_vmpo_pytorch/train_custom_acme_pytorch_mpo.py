import argparse
import collections
import copy
import math
import random
import time
from typing import Tuple, cast, Dict, Union, Optional
import os
import json
import threading

import dm_env # type: ignore
from dm_control import suite # type: ignore

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import torch.distributions as dist

from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage # type: ignore
from tensordict import TensorDict # type: ignore

_MPO_FLOAT_EPSILON = 1e-8


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ClipToSpec(nn.Module):
    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        super().__init__()
        self.register_buffer("low", low)
        self.register_buffer("high", high)

    def forward(self, x):
        return x.clamp(self.low, self.high)


class CriticMultiplexer(nn.Module):
    def __init__(
        self, obs_net: nn.Module | None, act_net: nn.Module | None, critic: nn.Module
    ):
        super().__init__()
        self.obs_net = obs_net
        self.act_net = act_net
        self.critic = critic

    def forward(self, obs, act):
        if self.obs_net:
            obs = self.obs_net(obs)
        if self.act_net:
            act = self.act_net(act)
        return self.critic(torch.cat([obs, act], dim=-1))


class LayerNormMLP(nn.Module):
    """MLP with LayerNorm on first hidden layer and ELU activations by default."""

    def __init__(
        self, input_dim: int, layer_sizes: Tuple[int, ...], activate_final: bool = False
    ):
        super().__init__()
        assert len(layer_sizes) >= 1
        self.first = nn.Linear(input_dim, layer_sizes[0])
        self.ln = nn.LayerNorm(layer_sizes[0])
        self.rest = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.rest.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.activate_final = activate_final
        self.activation = F.elu
        self.apply(init_weights)

    def forward(self, x):
        x = self.first(x)
        x = self.ln(x)
        x = torch.tanh(x)
        for i, layer in enumerate(self.rest):
            x = layer(x)
            if i < len(self.rest) - 1 or self.activate_final:
                x = self.activation(x)
        return x


class MultivariateNormalDiagHead(nn.Module):
    """Produces mean and scale for a diagonal Gaussian."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        init_scale: float,
        min_scale: float = 1e-6,
        tanh_mean: bool = False,
        fixed_scale: bool = False,
        use_independent: bool = True,
    ):
        super().__init__()
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.fixed_scale = fixed_scale
        self.tanh_mean = tanh_mean
        self.init_scale = init_scale
        self.min_scale = min_scale
        self.use_independent = use_independent
        self._fixed_scale = None
        if not fixed_scale:
            # output positive scale via softplus of a linear layer
            self.log_scale_layer = nn.Linear(input_dim, action_dim)
        else:
            self.register_buffer("_fixed_scale", torch.tensor(init_scale))

        # make mean bias zero for stable initial means
        with torch.no_grad():
            nn.init.uniform_(self.mean_layer.weight, a=-1e-4, b=1e-4)
            if self.mean_layer.bias is not None:
                self.mean_layer.bias.zero_()
            if not fixed_scale:
                nn.init.uniform_(self.log_scale_layer.weight, a=-1e-4, b=1e-4)
                if self.log_scale_layer.bias is not None:
                    self.log_scale_layer.bias.zero_()
            if self.mean_layer.bias is not None:
                self.mean_layer.bias.zero_()

    def _make_dist(self, mean: torch.Tensor, scale: torch.Tensor) -> dist.Distribution:
        if self.use_independent:
            return dist.Independent(dist.Normal(loc=mean, scale=scale), 1)
        else:
            cov = torch.diag_embed(scale.pow(2))
            return dist.MultivariateNormal(loc=mean, covariance_matrix=cov)

    def forward(self, inputs):
        mean = self.mean_layer(inputs)
        if self.tanh_mean:
            mean = torch.tanh(mean)
        if self._fixed_scale is not None:
            scale = torch.ones_like(mean) * float(self._fixed_scale)
        else:
            log_scale = self.log_scale_layer(inputs)
            scale = F.softplus(log_scale)
            zero = torch.zeros(1, device=scale.device)
            scale = scale * (self.init_scale / F.softplus(zero)) + self.min_scale
        return mean, scale


class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, layer_sizes=(512, 512, 256)):
        super().__init__()
        layers = []
        last = input_dim
        for s in layer_sizes:
            layers.append(nn.Linear(last, s))
            last = s
        layers.append(nn.Linear(last, 1))
        self.net = nn.ModuleList(layers)
        self.apply(init_weights)

        # Initialize final linear layer near zero (match TF NearZeroInitializedLinear).
        with torch.no_grad():
            final = cast(nn.Linear, self.net[-1])
            # Small uniform init around zero and zero bias.
            nn.init.uniform_(final.weight, a=-1e-4, b=1e-4)
            if final.bias is not None:
                nn.init.zeros_(final.bias)

    def forward(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i < len(self.net) - 1:
                x = F.elu(x)
        return x.squeeze(-1)


def _diag_normal_kl(
    p_mean: torch.Tensor,
    p_std: torch.Tensor,
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
    per_dim: bool,
) -> torch.Tensor:
    """
    KL( N(p_mean, p_std^2) || N(q_mean, q_std^2) ) for diagonal Gaussians.

    Shapes:
        p_mean, p_std, q_mean, q_std: [B, D]
    If per_dim = True:
        return [B, D] (no sum over D)
    If per_dim = False:
        return [B]    (sum over D)
    """
    # all shapes [B, D]
    var_p = p_std.pow(2)
    var_q = q_std.pow(2)
    diff = q_mean - p_mean

    # Element-wise KL
    # KL = log(σ_q/σ_p) + (σ_p^2 + (μ_p-μ_q)^2) / (2 σ_q^2) - 1/2
    log_term = torch.log(q_std / p_std)
    frac_term = (var_p + diff.pow(2)) / (2.0 * var_q)
    kl_elem = log_term + frac_term - 0.5  # [B, D]

    if per_dim:
        return kl_elem  # [B, D]
    else:
        return kl_elem.sum(dim=-1)  # [B]


def compute_weights_and_temperature_loss(
    q_values: torch.Tensor,  # [N, B]
    epsilon: float,
    temperature: torch.Tensor,  # scalar > 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match TensorFlow compute_weights_and_temperature_loss exactly:
      - temper with temperature (no grad to q_values),
      - softmax over axis=0 (action samples),
      - detach normalized weights,
      - temperature dual loss computed like TF and multiplied by temperature.

    This version ensures temperature is not re-created/detached and constants
    are created using temperature.new_tensor(...) for consistent dtype/device.
    """
    # Temper the given Q-values using the current temperature; stop gradient on q_values.
    tempered_q_values = q_values.detach() / temperature

    # Compute normalized importance weights across action-samples axis (N).
    normalized_weights = torch.softmax(tempered_q_values, dim=0).detach()

    # Temperature loss per TF: epsilon + mean(logsumexp(tempered_q)) - log(num_actions)
    q_logsumexp = torch.logsumexp(tempered_q_values, dim=0)  # [B]
    num_actions = float(q_values.shape[0])
    # Create tensors matching temperature's dtype/device without breaking autograd.
    log_num_actions = temperature.new_tensor(math.log(num_actions))
    eps_t = temperature.new_tensor(float(epsilon))
    loss_temperature_inner = eps_t + q_logsumexp.mean() - log_num_actions
    loss_temperature = temperature * loss_temperature_inner

    return normalized_weights, loss_temperature


def compute_nonparametric_kl_from_normalized_weights(
    normalized_weights: torch.Tensor,  # [N, B]
) -> torch.Tensor:
    """
    Estimate the actualized KL between the non-parametric and target policies,
    matching the TF implementation:
      integrand = log(N * w + eps)
      return sum_w w * integrand  (expectation over non-parametric policy)
    """
    num_action_samples = float(normalized_weights.shape[0])
    integrand = torch.log(num_action_samples * normalized_weights + _MPO_FLOAT_EPSILON)
    return (normalized_weights * integrand).sum(dim=0)  # [B]


def compute_cross_entropy_loss(
    sampled_actions: torch.Tensor,  # [N, B, D]
    normalized_weights: torch.Tensor,  # [N, B]
    online_action_distribution: dist.Distribution,  # Independent(Normal) or Multivariate
) -> torch.Tensor:
    """
    Compute the cross-entropy loss equivalent to TF:
      - log_prob has shape [N, B]
      - weighted sum over N then mean over batch B
    """
    log_prob = online_action_distribution.log_prob(sampled_actions)  # [N, B]
    loss_policy_gradient = -torch.sum(log_prob * normalized_weights, dim=0)  # [B]
    return loss_policy_gradient.mean()  # scalar


def compute_parametric_kl_penalty_and_dual_loss(
    kl: torch.Tensor,  # [B, D] or [B]
    alpha: torch.Tensor,  # post-softplus alpha (D,) or (1,)
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match TF exactly:
      mean_kl = reduce_mean(kl, axis=0) -> shape [D] or [1]
      loss_kl = sum(stop_gradient(alpha) * mean_kl)
      loss_alpha = sum(alpha * (epsilon - stop_gradient(mean_kl)))
    """
    mean_kl = kl.mean(dim=0)  # [D] or [1]
    loss_kl = (alpha.detach() * mean_kl).sum()
    loss_alpha = (alpha * (epsilon - mean_kl.detach())).sum()
    return loss_kl, loss_alpha


class MPOLoss(nn.Module):
    """
    PyTorch MPOLoss that mirrors the TensorFlow MPOLoss precisely in:
      - creation and clamping of log-dual vars,
      - softplus transform to get positive duals,
      - E-step temperature computation with stop_gradient semantics,
      - optional MO-MPO action penalty (identical operations and combination),
      - decomposition into fixed-mean / fixed-stddev distributions,
      - computation of cross-entropy losses, parametric KL penalties and dual losses,
      - same returned diagnostics keys (names are PyTorch style but semantics match).
    """

    def __init__(
        self,
        epsilon: float,
        epsilon_mean: float,
        epsilon_stddev: float,
        init_log_temperature: float,
        init_log_alpha_mean: float,
        init_log_alpha_stddev: float,
        action_dim: int,
        per_dim_constraining: bool = True,
        action_penalization: bool = True,
        epsilon_penalty: float = 0.001,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self._epsilon = float(epsilon)
        self._epsilon_mean = float(epsilon_mean)
        self._epsilon_stddev = float(epsilon_stddev)
        self._epsilon_penalty = float(epsilon_penalty)
        self._per_dim_constraining = per_dim_constraining
        self._action_penalization = action_penalization

        # register epsilons (buffers) so they move with .to(...)
        self.register_buffer(
            "epsilon_buf", torch.tensor(self._epsilon, dtype=dtype, device=device)
        )
        self.register_buffer(
            "epsilon_mean_buf",
            torch.tensor(self._epsilon_mean, dtype=dtype, device=device),
        )
        self.register_buffer(
            "epsilon_stddev_buf",
            torch.tensor(self._epsilon_stddev, dtype=dtype, device=device),
        )
        if self._action_penalization:
            self.register_buffer(
                "epsilon_penalty_buf",
                torch.tensor(self._epsilon_penalty, dtype=dtype, device=device),
            )

        # Duals in log-space (trainable)
        self.log_temperature = nn.Parameter(
            torch.tensor([init_log_temperature], dtype=dtype, device=device)
        )
        alpha_shape = (action_dim,) if per_dim_constraining else (1,)
        self.log_alpha_mean = nn.Parameter(
            torch.full(alpha_shape, init_log_alpha_mean, dtype=dtype, device=device)
        )
        self.log_alpha_stddev = nn.Parameter(
            torch.full(alpha_shape, init_log_alpha_stddev, dtype=dtype, device=device)
        )
        if self._action_penalization:
            self._log_penalty_temperature = nn.Parameter(
                torch.tensor([init_log_temperature], dtype=dtype, device=device)
            )

        self.min_log_temperature = torch.tensor(-18.0, dtype=dtype, device=device)
        self.min_log_alpha = torch.tensor(-18.0, dtype=dtype, device=device)

    def forward(
        self,
        actions: torch.Tensor,  # [N, B, D]
        q_values: torch.Tensor,  # [N, B]
        online_mean: torch.Tensor,  # [B, D]
        online_scale: torch.Tensor,  # [B, D]
        target_mean: torch.Tensor,  # [B, D]
        target_scale: torch.Tensor,  # [B, D]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # dtype convenience
        dtype = q_values.dtype

        # Project dual variables' log-values (in-place, no grads).
        with torch.no_grad():
            self.log_temperature.data = torch.maximum(
                self.log_temperature.data, self.min_log_temperature.to(dtype)
            )
            self.log_alpha_mean.data = torch.maximum(
                self.log_alpha_mean.data, self.min_log_alpha.to(dtype)
            )
            self.log_alpha_stddev.data = torch.maximum(
                self.log_alpha_stddev.data, self.min_log_alpha.to(dtype)
            )
            if self._action_penalization:
                self._log_penalty_temperature.data = torch.maximum(
                    self._log_penalty_temperature.data,
                    self.min_log_temperature.to(dtype),
                )

        # Transform dual variables from log-space using softplus, add epsilon for safety.
        temperature = F.softplus(self.log_temperature) + _MPO_FLOAT_EPSILON  # [1]
        alpha_mean = F.softplus(self.log_alpha_mean) + _MPO_FLOAT_EPSILON  # [D] or [1]
        alpha_stddev = (
            F.softplus(self.log_alpha_stddev) + _MPO_FLOAT_EPSILON
        )  # [D] or [1]

        # E-step: compute normalized weights & temperature loss (detach semantics inside helper)
        normalized_weights, loss_temperature = compute_weights_and_temperature_loss(
            q_values=q_values, epsilon=self._epsilon, temperature=temperature
        )

        # Diagnostic: KL between non-parametric and target
        kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
            normalized_weights
        )
        penalty_kl_nonparametric = None

        # Optional MO-MPO penalty
        if self._action_penalization:
            penalty_temperature = (
                F.softplus(self._log_penalty_temperature) + _MPO_FLOAT_EPSILON
            )
            diff_out_of_bound = actions - actions.clamp(-1.0, 1.0)  # [N,B,D]
            cost_out_of_bound = -diff_out_of_bound.norm(dim=-1)  # [N,B]

            penalty_normalized_weights, loss_penalty_temperature = (
                compute_weights_and_temperature_loss(
                    q_values=cost_out_of_bound,
                    epsilon=self._epsilon_penalty,
                    temperature=penalty_temperature,
                )
            )
            penalty_kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
                penalty_normalized_weights
            )

            # Combine weights and temperature losses just like TF.
            normalized_weights = normalized_weights + penalty_normalized_weights
            loss_temperature = loss_temperature + loss_penalty_temperature

        # Decompose online policy into fixed-stddev and fixed-mean distributions (matching TF)
        fixed_stddev_dist = dist.Independent(
            dist.Normal(loc=online_mean, scale=target_scale), 1
        )
        fixed_mean_dist = dist.Independent(
            dist.Normal(loc=target_mean, scale=online_scale), 1
        )

        # Cross-entropy (M-step) terms
        loss_policy_mean = compute_cross_entropy_loss(
            actions, normalized_weights, fixed_stddev_dist
        )
        loss_policy_stddev = compute_cross_entropy_loss(
            actions, normalized_weights, fixed_mean_dist
        )

        # KL computations: target || fixed component (per-dim or aggregated)
        per_dim = self._per_dim_constraining
        kl_mean = _diag_normal_kl(
            p_mean=target_mean,
            p_std=target_scale,
            q_mean=fixed_stddev_dist.base_dist.loc,
            q_std=fixed_stddev_dist.base_dist.scale,
            per_dim=per_dim,
        )
        kl_stddev = _diag_normal_kl(
            p_mean=target_mean,
            p_std=target_scale,
            q_mean=fixed_mean_dist.base_dist.loc,
            q_std=fixed_mean_dist.base_dist.scale,
            per_dim=per_dim,
        )

        # Parametric KL penalties and dual losses (alpha adaptation)
        loss_kl_mean, loss_alpha_mean = compute_parametric_kl_penalty_and_dual_loss(
            kl=kl_mean, alpha=alpha_mean, epsilon=self._epsilon_mean
        )
        loss_kl_stddev, loss_alpha_stddev = compute_parametric_kl_penalty_and_dual_loss(
            kl=kl_stddev, alpha=alpha_stddev, epsilon=self._epsilon_stddev
        )

        # Combine everything exactly like TF
        loss_policy = loss_policy_mean + loss_policy_stddev
        loss_kl_penalty = loss_kl_mean + loss_kl_stddev
        loss_dual = loss_alpha_mean + loss_alpha_stddev + loss_temperature
        loss = loss_policy + loss_kl_penalty + loss_dual

        # Prepare diagnostics similar to TF names/semantics
        stats: Dict[str, torch.Tensor] = {}
        stats["dual_alpha_mean"] = alpha_mean.mean()
        stats["dual_alpha_stddev"] = alpha_stddev.mean()
        stats["dual_temperature"] = temperature.mean()
        stats["loss_policy"] = loss_policy.detach()
        stats["total_loss"] = loss.detach()
        stats["loss_alpha"] = (loss_alpha_mean + loss_alpha_stddev).detach()
        stats["loss_temperature"] = loss_temperature.detach()
        stats["kl_q_rel"] = kl_nonparametric.mean() / self._epsilon
        if self._action_penalization and penalty_kl_nonparametric is not None:
            stats["penalty_kl_q_rel"] = (
                penalty_kl_nonparametric.mean() / self._epsilon_penalty
            )
        stats["kl_mean_rel"] = kl_mean.mean() / self._epsilon_mean
        stats["kl_stddev_rel"] = kl_stddev.mean() / self._epsilon_stddev

        if per_dim:
            kl_mean_batch_mean = kl_mean.mean(dim=0)
            kl_stddev_batch_mean = kl_stddev.mean(dim=0)
            stats["kl_mean_rel_min"] = kl_mean_batch_mean.min() / self._epsilon_mean
            stats["kl_mean_rel_max"] = kl_mean_batch_mean.max() / self._epsilon_mean
            stats["kl_stddev_rel_min"] = (
                kl_stddev_batch_mean.min() / self._epsilon_stddev
            )
            stats["kl_stddev_rel_max"] = (
                kl_stddev_batch_mean.max() / self._epsilon_stddev
            )

        stats["q_min"] = q_values.min(dim=0).values.mean()
        stats["q_max"] = q_values.max(dim=0).values.mean()
        pi_stddev = online_scale
        stats["pi_stddev_min"] = pi_stddev.min(dim=-1).values.mean()
        stats["pi_stddev_max"] = pi_stddev.max(dim=-1).values.mean()
        stats["pi_stddev_cond"] = (
            pi_stddev.max(dim=-1).values
            / (pi_stddev.min(dim=-1).values + _MPO_FLOAT_EPSILON)
        ).mean()

        return loss, stats


class NStepAccumulator:
    __slots__ = ("n_step", "gamma", "_traj")

    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self._traj = collections.deque()

    def push(self, obs, action, reward, discount, next_obs, done):
        self._traj.append((obs, action, reward, discount, next_obs, done))
        ready = []
        while len(self._traj) >= self.n_step or (self._traj and self._traj[0][5]):
            ret = 0.0
            cur_discount = 1.0
            next_o = None
            terminal = False
            for idx, (_, _, r, d, nobs, dn) in enumerate(self._traj):
                ret += cur_discount * r
                cur_discount *= self.gamma * d
                next_o = nobs
                terminal = dn
                if idx + 1 >= self.n_step:
                    break
            obs0, act0, _, _, _, _ = self._traj[0]
            ready.append((obs0, act0, ret, cur_discount, next_o, terminal))
            self._traj.popleft()
            if not self._traj:
                break
        return ready


class MPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low,
        action_high,
        device: torch.device,
        lr_dual: float,
        min_replay_size: int,
        max_replay_size: int,
        policy_hidden=(256, 256, 256),
        critic_hidden=(512, 512, 256),
        init_scale=0.7,
        gamma=0.99,
        n_step=5,
        batch_size=256,
        num_samples=20,
        target_policy_update_period=25,
        target_critic_update_period=100,
        lr_policy=1e-4,
        lr_critic=1e-4,
        clip_norm: float = 40.0,
        action_penalization=True,
        per_dim=True,
        samples_per_insert=32.0,
        ratio_tolerance=0.1,
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.target_policy_update_period = target_policy_update_period
        self.target_critic_update_period = target_critic_update_period
        # clip_norm > 0 enables gradient clipping to that max-norm value.
        self.clip_norm = float(clip_norm)
        self.clipping_enabled = self.clip_norm > 0.0

        # new: store configurable minimum replay size for learning
        self.min_replay_size = int(min_replay_size)

        # observation encoder used by critic (trained by critic)
        self.obs_encoder = LayerNormMLP(
            obs_dim, tuple(policy_hidden), activate_final=True
        ).to(device)

        # policy head: maps embeddings -> action distribution (trained by policy optimizer)
        self.policy_head = MultivariateNormalDiagHead(
            policy_hidden[-1], action_dim, init_scale=init_scale
        ).to(device)

        # critic: input = obs_embedding + action (embedding from obs_encoder)
        critic_input_dim = policy_hidden[-1] + action_dim

        self.clip_to_spec = ClipToSpec(
            torch.tensor(action_low, dtype=torch.float32, device=device),
            torch.tensor(action_high, dtype=torch.float32, device=device),
        )
        self.critic = CriticMultiplexer(
            obs_net=self.obs_encoder,
            act_net=self.clip_to_spec,
            critic=CriticNetwork(critic_input_dim, layer_sizes=critic_hidden).to(
                device
            ),
        )

        # target networks (hard copy)
        self.target_obs_encoder = copy.deepcopy(self.obs_encoder).to(device)
        self.target_policy_head = copy.deepcopy(self.policy_head).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.target_critic.obs_net = self.target_obs_encoder

        # --- NEW: value head (V) and its target/optimizer ---
        # light-weight scalar head that maps obs embedding -> value
        self.value_head = nn.Linear(policy_hidden[-1], 1).to(device)
        # small init around zero for stability
        with torch.no_grad():
            nn.init.uniform_(self.value_head.weight, a=-1e-4, b=1e-4)
            if self.value_head.bias is not None:
                self.value_head.bias.zero_()
        self.target_value_head = copy.deepcopy(self.value_head).to(device)

        self.mpo_loss = MPOLoss(
            action_dim=action_dim,
            per_dim_constraining=per_dim,
            action_penalization=action_penalization,
            device=device,
            epsilon=1e-1,
            epsilon_penalty=1e-3,
            epsilon_mean=2.5e-3,
            epsilon_stddev=1e-6,
            init_log_temperature=10.0,
            init_log_alpha_mean=10.0,
            init_log_alpha_stddev=1000.0,
        ).to(device)

        self._critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self._policy_optimizer = optim.Adam(self.policy_head.parameters(), lr=lr_policy)
        self._dual_optimizer = optim.Adam(self.mpo_loss.parameters(), lr=lr_dual)
        # new value optimizer (reuse critic lr by default)
        self._value_optimizer = optim.Adam(self.value_head.parameters(), lr=lr_critic)

        self.replay = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=max_replay_size),
            batch_size=batch_size,
        )
        self._nstep_accumulator = NStepAccumulator(n_step=n_step, gamma=gamma)
        self._learn_steps = 0
        self.samples_per_insert = samples_per_insert
        self.ratio_tolerance = ratio_tolerance
        self._num_inserts = 0
        self._num_samples = 0

    def copy_policy_to(self, obs_encoder: nn.Module, policy_head: nn.Module) -> None:
        obs_encoder.load_state_dict(self.obs_encoder.state_dict())
        policy_head.load_state_dict(self.policy_head.state_dict())

    def select_action(self, obs: np.ndarray, stochastic: bool = True) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            emb = self.obs_encoder(obs_t)
            mean, scale = self.policy_head(emb)
            if stochastic:
                action = mean + scale * torch.randn_like(mean)
            else:
                action = mean
            action = torch.clamp(action.squeeze(0), self.action_low, self.action_high)
        return action.cpu().numpy()

    def store_transition(self, obs, action, reward, discount, next_obs, done):
        ready = self._nstep_accumulator.push(
            obs, action, reward, discount, next_obs, done
        )
        for obs_i, act_i, ret_i, disc_i, next_obs_i, done_i in ready:
            data = TensorDict(
                {
                    "obs": torch.tensor(obs_i, dtype=torch.float32),
                    "action": torch.tensor(act_i, dtype=torch.float32),
                    "reward": torch.tensor(ret_i, dtype=torch.float32),
                    "discount": torch.tensor(disc_i, dtype=torch.float32),
                    "next_obs": torch.tensor(next_obs_i, dtype=torch.float32),
                    "done": torch.tensor(done_i, dtype=torch.bool),
                },
                batch_size=[],
            )
            self.replay.add(data)
            self._num_inserts += 1

    def _can_sample(self):
        replay_len = len(self.replay)
        if replay_len < max(self.min_replay_size, self.batch_size):
            return False
        if self.samples_per_insert is None:
            return True
        target = (
            self.samples_per_insert + self.ratio_tolerance * self.samples_per_insert
        )
        return self._num_samples <= target * max(1, self._num_inserts)

    def learn_step(self):
        if not self._can_sample():
            return None

        batch = self.replay.sample()
        self._num_samples += 1

        device = self.device
        obs_b = batch["obs"].to(device)
        actions_b = batch["action"].to(device)
        rewards_b = batch["reward"].to(device)
        discounts_b = batch["discount"].to(device)
        next_obs_b = batch["next_obs"].to(device)

        # --- compute target Q via target policy sampling ---
        with torch.no_grad():
            emb_next = self.target_obs_encoder(next_obs_b)
            t_mean, t_scale = self.target_policy_head(emb_next)  # (B,D)
            N = self.num_samples
            # sample N actions per batch element
            t_mean_exp = t_mean.unsqueeze(0).expand(N, -1, -1)  # (N,B,D)
            t_scale_exp = t_scale.unsqueeze(0).expand(N, -1, -1)
            eps = torch.randn_like(t_mean_exp)
            sampled_actions = t_mean_exp + t_scale_exp * eps  # (N,B,D)

            # Target critic expects raw obs; its obs_net will encode.
            obs_next_exp = (
                next_obs_b.unsqueeze(0)
                .expand(N, -1, -1)
                .reshape(N * self.batch_size, -1)
            )
            sampled_actions_resh = sampled_actions.reshape(N * self.batch_size, -1)
            q_samples = self.target_critic(obs_next_exp, sampled_actions_resh).view(
                N, self.batch_size
            )  # (N,B)

            # guard against NaN / Inf in target Qs
            q_samples = torch.nan_to_num(q_samples, nan=0.0, posinf=1e6, neginf=-1e6)
            q_t = q_samples.mean(dim=0)  # (B,)

            # Compute scalar summaries for q (match TF learner logging):
            # q_min := mean over batch of min over action-samples
            # q_max := mean over batch of max over action-samples
            q_min = float(q_samples.min(dim=0).values.mean().item())
            q_max = float(q_samples.max(dim=0).values.mean().item())

        # critic loss (TD)
        target = rewards_b + discounts_b * q_t

        q_tm1 = self.critic(obs_b, actions_b)

        # Match Acme/td_learning: TD error = target - v_tm1, loss = 0.5 * td_error^2
        td_error = target - q_tm1
        critic_loss = 0.5 * (td_error**2).mean()
        # small diagnostics for logging
        td_error_mean = float(td_error.abs().mean().item())
        target_mean = float(target.mean().item())

        # --- NEW: V-MPO value baseline and advantage computation ---
        # Compute baseline V(next_obs) using online value head (detach embedding to avoid
        # updating encoder here; we only update value_head weights).
        with torch.no_grad():
            # Ensure MPO dual log vars are clamped like MPOLoss would (so temperature computed below matches MPOLoss)
            self.mpo_loss.log_temperature.data = torch.maximum(
                self.mpo_loss.log_temperature.data,
                self.mpo_loss.min_log_temperature.to(
                    self.mpo_loss.log_temperature.dtype
                ),
            )
            self.mpo_loss.log_alpha_mean.data = torch.maximum(
                self.mpo_loss.log_alpha_mean.data,
                self.mpo_loss.min_log_alpha.to(self.mpo_loss.log_alpha_mean.dtype),
            )
            self.mpo_loss.log_alpha_stddev.data = torch.maximum(
                self.mpo_loss.log_alpha_stddev.data,
                self.mpo_loss.min_log_alpha.to(self.mpo_loss.log_alpha_stddev.dtype),
            )

        # baseline computed from obs_encoder -> value_head; detach encoder embedding so value update only affects value_head
        emb_for_value_detached = self.obs_encoder(next_obs_b).detach()
        v_baseline = (
            self.value_head(emb_for_value_detached).squeeze(-1).detach()
        )  # (B,)

        # advantages across sampled actions (N,B)
        advantages = q_samples - v_baseline.unsqueeze(0)

        # Compute non-parametric weights using MPO's temperature (so value target is consistent)
        with torch.no_grad():
            temperature = F.softplus(self.mpo_loss.log_temperature) + _MPO_FLOAT_EPSILON
            normalized_weights, _loss_temp = compute_weights_and_temperature_loss(
                q_values=advantages,
                epsilon=self.mpo_loss._epsilon,
                temperature=temperature,
            )
            # weighted target value for V update
            target_v = (normalized_weights * q_samples).sum(dim=0)  # (B,)

        # Update value head towards weighted Q-target (MSE). Detach target_v to avoid leakage.
        self._value_optimizer.zero_grad()
        # Use detached encoder embedding again to avoid interfering with critic's encoder training here.
        pred_v = self.value_head(emb_for_value_detached).squeeze(-1)
        value_loss = 0.5 * ((pred_v - target_v.detach()) ** 2).mean()
        value_loss.backward()
        self._value_optimizer.step()

        # --- end value update ---

        # Update critic (as before)
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clipping_enabled:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_norm)
        self._critic_optimizer.step()

        # Compute policy head inputs (use detached encoder embedding; policy_head is updated)
        emb_for_policy = self.obs_encoder(next_obs_b).detach()
        online_mean, online_scale = self.policy_head(emb_for_policy)

        # compute MPO loss with advantages (V-MPO uses advantage-weighted E-step)
        policy_loss, policy_stats = self.mpo_loss(
            sampled_actions, advantages, online_mean, online_scale, t_mean, t_scale
        )

        self._policy_optimizer.zero_grad()
        self._dual_optimizer.zero_grad()
        policy_loss.backward()
        if self.clipping_enabled:
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_head.parameters()), self.clip_norm
            )
        self._policy_optimizer.step()
        self._dual_optimizer.step()

        self._learn_steps += 1

        # periodic target hard updates
        # Update target policy head (periodic).
        if self._learn_steps % self.target_policy_update_period == 0:
            self.target_policy_head.load_state_dict(self.policy_head.state_dict())
        # Update target critic and the target observation encoder (periodic).
        if self._learn_steps % self.target_critic_update_period == 0:
            self.target_critic.critic.load_state_dict(self.critic.critic.state_dict())
            self.target_obs_encoder.load_state_dict(self.obs_encoder.state_dict())
            # --- NEW: sync target value head ---
            self.target_value_head.load_state_dict(self.value_head.state_dict())

        # Prepare diagnostics similar to TF names/semantics
        stats: Dict[str, torch.Tensor] = {}
        stats["dual_alpha_mean"] = alpha_mean.mean()
        stats["dual_alpha_stddev"] = alpha_stddev.mean()
        stats["dual_temperature"] = temperature.mean()
        stats["loss_policy"] = loss_policy.detach()
        stats["total_loss"] = loss.detach()
        stats["loss_alpha"] = (loss_alpha_mean + loss_alpha_stddev).detach()
        stats["loss_temperature"] = loss_temperature.detach()
        stats["kl_q_rel"] = kl_nonparametric.mean() / self._epsilon
        if self._action_penalization and penalty_kl_nonparametric is not None:
            stats["penalty_kl_q_rel"] = (
                penalty_kl_nonparametric.mean() / self._epsilon_penalty
            )
        stats["kl_mean_rel"] = kl_mean.mean() / self._epsilon_mean
        stats["kl_stddev_rel"] = kl_stddev.mean() / self._epsilon_stddev

        if per_dim:
            kl_mean_batch_mean = kl_mean.mean(dim=0)
            kl_stddev_batch_mean = kl_stddev.mean(dim=0)
            stats["kl_mean_rel_min"] = kl_mean_batch_mean.min() / self._epsilon_mean
            stats["kl_mean_rel_max"] = kl_mean_batch_mean.max() / self._epsilon_mean
            stats["kl_stddev_rel_min"] = (
                kl_stddev_batch_mean.min() / self._epsilon_stddev
            )
            stats["kl_stddev_rel_max"] = (
                kl_stddev_batch_mean.max() / self._epsilon_stddev
            )

        stats["q_min"] = q_values.min(dim=0).values.mean()
        stats["q_max"] = q_values.max(dim=0).values.mean()
        pi_stddev = online_scale
        stats["pi_stddev_min"] = pi_stddev.min(dim=-1).values.mean()
        stats["pi_stddev_max"] = pi_stddev.max(dim=-1).values.mean()
        stats["pi_stddev_cond"] = (
            pi_stddev.max(dim=-1).values
            / (pi_stddev.min(dim=-1).values + _MPO_FLOAT_EPSILON)
        ).mean()

        return loss, stats


class EnvironmentLoop:
    """Run episodes for an actor-like object (select_action/observe_first/observe/update/post_step).

    The actor must expose:
      - observe_first(obs)
      - select_action(obs) -> action (numpy)
      - observe(prev_obs, action, reward, next_obs, done)
      - update(total_steps)
      - post_step() -> int  (increments/returns global step or returns current step (for evaluator))
      - label: str  (used for logging)
    """

    def __init__(self, environment, actor, shared_state):
        self._environment = environment
        self._actor = actor
        self._shared_state = shared_state
        self._episodes = 0
        self._total_steps = 0

    def _run_episode(self):
        episode_return = 0.0
        episode_steps = 0

        obs, _ = self._environment.reset()
        obs_flat = flatten_observation(obs)

        self._actor.observe_first(obs_flat)

        done = False
        start_time = time.time()
        while not done:
            action = self._actor.select_action(obs_flat)
            next_obs, reward, terminated, truncated, _ = self._environment.step(action)
            done = bool(terminated or truncated)
            next_flat = flatten_observation(next_obs)

            # Let actor process transition (e.g., insert into replay)
            self._actor.observe(obs_flat, action, float(reward), next_flat, done)

            # Update/ sync actor as necessary (policy syncs etc.)
            # post_step should increment global step (for actors) or return current step (for evaluator).
            total_steps = self._actor.post_step()
            self._actor.update(total_steps)

            # Synchronously run learning in-actor to match original MPO (actor+learner combined).
            # This will call agent.learn_step repeatedly until replay isn't ready.
            if hasattr(self._actor, "maybe_learn"):
                self._actor.maybe_learn(self._actor.max_learn_steps_per_call)

            # Trigger evaluation periodically (actor has .evaluator)
            if hasattr(self._actor, "evaluator") and getattr(
                self._actor, "eval_interval_steps", None
            ):
                steps = int(self._shared_state.get("steps", 0))
                if steps > 0 and steps % self._actor.eval_interval_steps == 0:
                    self._actor.evaluator.run()

            episode_return += float(reward)
            episode_steps += 1
            obs_flat = next_flat

            if self._actor.stop_event.is_set():
                break

        duration = time.time() - start_time
        result = {
            "episode_length": episode_steps,
            "episode_return": episode_return,
            "episode_duration": duration,
            "steps_per_second": episode_steps / duration if duration > 0 else 0.0,
        }
        return result

    def run(self, num_episodes: Optional[int] = None, num_steps: Optional[int] = None):
        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        def should_terminate(episodes_done, steps_done):
            if num_episodes is not None and episodes_done >= num_episodes:
                return True
            if num_steps is not None and steps_done >= num_steps:
                return True
            # also stop if actor requests it
            return self._actor.stop_event.is_set()

        episodes = 0
        steps_done = 0
        while not should_terminate(episodes, steps_done):
            episode_result = self._run_episode()
            self._episodes += 1
            episodes += 1
            steps_done += episode_result["episode_length"]
            self._episodes += 1
            self._total_steps += episode_result["episode_length"]
            episode_result["total_episodes"] = self._episodes
            episode_result["total_steps"] = self._total_steps
            wandb.log(
                {self._actor.label + "/" + k: v for k, v in episode_result.items()}
            )
        return (
            self._shared_state.get("steps", None)
            if self._shared_state is not None
            else steps_done
        )


class Actor:
    """Actor encapsulates environment interaction, policy syncing and replay insertion."""

    def __init__(
        self,
        agent: MPOAgent,
        env_name: str,
        device: torch.device,
        stop_event: threading.Event,
        shared_state: dict,
        policy_sync_interval: int,
        eval_interval_steps: int = 5000,
        max_learn_steps_per_call: int = 10,
    ):
        self.agent = agent
        self.env_name = env_name
        self.device = device
        self.stop_event = stop_event
        self.shared_state = shared_state
        self.policy_sync_interval = max(1, policy_sync_interval)
        self._actor_encoder = None
        self._actor_policy_head = None
        self._sync_counter = 0
        self.label = "actor"
        # evaluation / learning throttling params
        self.eval_interval_steps = int(eval_interval_steps)
        self.max_learn_steps_per_call = int(max_learn_steps_per_call)
        self.evaluator = Evaluator(
            agent=agent,
            env_name=env_name,
            device=device,
            stop_event=stop_event,
            shared_state=shared_state,
        )

    def setup_local_modules(self):
        """Create local copies of policy modules used by this actor thread."""
        self._actor_encoder = copy.deepcopy(self.agent.obs_encoder).to(self.device)
        self._actor_policy_head = copy.deepcopy(self.agent.policy_head).to(self.device)
        # sync initial params
        self.agent.copy_policy_to(self._actor_encoder, self._actor_policy_head)
        self._actor_encoder.eval()
        self._actor_policy_head.eval()
        self._sync_counter = 0

    def select_action(
        self, obs_flat: np.ndarray, stochastic: bool = True
    ) -> np.ndarray:
        # draw action from local modules
        if self._actor_encoder is None or self._actor_policy_head is None:
            self.setup_local_modules()
        if self._actor_encoder is not None and self._actor_policy_head is not None:
            return run_policy_modules(
                self._actor_encoder,
                self._actor_policy_head,
                obs_flat,
                self.device,
                self.agent.action_low,
                self.agent.action_high,
                stochastic=stochastic,
            )
        raise RuntimeError("Actor policy modules are not initialized.")

    def observe_first(self, obs_flat: np.ndarray):
        # no-op required by EnvironmentLoop interface
        return

    def observe(
        self,
        prev_obs_flat: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs_flat: np.ndarray,
        done: bool,
    ):
        # convert and store into replay using agent
        step_discount = 0.0 if done else 1.0
        self.agent.store_transition(
            prev_obs_flat, action, float(reward), step_discount, next_obs_flat, done
        )

    def update(self, total_steps: int):
        # sync local policy periodically
        self._sync_counter += 1
        if self._sync_counter >= self.policy_sync_interval:
            if self._actor_encoder is None or self._actor_policy_head is None:
                return
            self.agent.copy_policy_to(self._actor_encoder, self._actor_policy_head)
            self._actor_encoder.eval()
            self._actor_policy_head.eval()
            self._sync_counter = 0

    def post_step(self) -> int:
        # increment global step counter and return it
        self.shared_state["steps"] += 1
        return int(self.shared_state["steps"])

    def run(self, max_actor_steps: int):
        env = make_env(self.env_name)
        self.setup_local_modules()
        steps_left = max_actor_steps
        loop = EnvironmentLoop(
            environment=env,
            actor=self,
            shared_state=self.shared_state,
        )
        while steps_left > 0 and not self.stop_event.is_set():
            # run segments; evaluation is triggered from EnvironmentLoop based on shared_state
            seg = min(steps_left, max(1, self.eval_interval_steps))
            loop.run(num_steps=seg)
            steps_left -= seg
        self.stop_event.set()
        env.close()

    def maybe_learn(self, max_steps: int = 1):
        """Run up to max_steps learning iterations to avoid blocking the actor."""
        steps = 0
        while steps < max_steps and not self.stop_event.is_set():
            stats = self.agent.learn_step()
            if stats is None:
                break
            log_dict = {}
            for key, value in stats.items():
                if isinstance(value, torch.Tensor):
                    log_dict["learner/" + key] = float(value.detach().cpu().item())
                else:
                    log_dict["learner/" + key] = value
            log_dict["global_actor_step"] = int(self.shared_state.get("steps", 0))
            wandb.log(log_dict)
            steps += 1

    # New: collect steps without triggering learning (used for controlled data collection)
    def collect_steps(self, max_steps: int):
        """Collect up to max_steps environment interactions, inserting into replay but
        without invoking learner or evaluator. Returns number of env steps collected."""
        env = make_env(self.env_name)
        self.setup_local_modules()
        steps_left = max_steps
        steps_collected = 0
        obs, _ = env.reset()
        obs_flat = flatten_observation(obs)
        # observe_first is a no-op but keep call for symmetry
        self.observe_first(obs_flat)
        while steps_left > 0 and not self.stop_event.is_set():
            action = self.select_action(obs_flat)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            next_flat = flatten_observation(next_obs)
            # insert transition (agent handles n-step accumulation)
            self.observe(obs_flat, action, float(reward), next_flat, done)
            # increment global step
            self.post_step()
            # periodic local policy sync (same logic as update uses a counter; emulate here)
            self._sync_counter += 1
            if self._sync_counter >= self.policy_sync_interval:
                if (
                    self._actor_encoder is not None
                    and self._actor_policy_head is not None
                ):
                    self.agent.copy_policy_to(
                        self._actor_encoder, self._actor_policy_head
                    )
                    self._actor_encoder.eval()
                    self._actor_policy_head.eval()
                self._sync_counter = 0
            obs_flat = next_flat
            steps_left -= 1
            steps_collected += 1
            if done:
                obs, _ = env.reset()
                obs_flat = flatten_observation(obs)
        env.close()
        return steps_collected


class Evaluator:
    """Evaluator runs continuous evaluation episodes and logs results."""

    def __init__(
        self,
        agent: MPOAgent,
        env_name: str,
        device: torch.device,
        stop_event: threading.Event,
        shared_state: dict,
    ):
        self.agent = agent
        self.env_name = env_name
        self.device = device
        self.stop_event = stop_event
        self.shared_state = shared_state

        # local eval modules (created and synced as needed)
        self._eval_encoder = None
        self._eval_policy_head = None
        self.label = "eval"

    def setup_local_modules(self):
        self._eval_encoder = copy.deepcopy(self.agent.obs_encoder).to(self.device)
        self._eval_policy_head = copy.deepcopy(self.agent.policy_head).to(self.device)
        self.agent.copy_policy_to(self._eval_encoder, self._eval_policy_head)
        self._eval_encoder.eval()
        self._eval_policy_head.eval()

    def select_action(
        self, obs_flat: np.ndarray, stochastic: bool = False
    ) -> np.ndarray:
        if self._eval_encoder is None or self._eval_policy_head is None:
            self.setup_local_modules()
        assert (
            self._eval_encoder is not None and self._eval_policy_head is not None
        ), "Evaluation modules must be initialized"
        return run_policy_modules(
            self._eval_encoder,
            self._eval_policy_head,
            obs_flat,
            self.device,
            self.agent.action_low,
            self.agent.action_high,
            stochastic=stochastic,
        )

    def observe_first(self, obs_flat: np.ndarray):
        # no-op for evaluator
        return

    def observe(
        self,
        prev_obs_flat: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs_flat: np.ndarray,
        done: bool,
    ):
        # evaluator doesn't insert into replay
        return

    def update(self, total_steps: int):
        # Sync policy weights periodically based on total_steps (simple policy copy every N steps).
        # Use agent.target intervals for a reasonable sync cadence.
        update_period = max(1, self.agent.target_policy_update_period)
        if total_steps % update_period == 0:
            # refresh local modules from agent
            if self._eval_encoder is None or self._eval_policy_head is None:
                self.setup_local_modules()
            else:
                self.agent.copy_policy_to(self._eval_encoder, self._eval_policy_head)
                self._eval_encoder.eval()
                self._eval_policy_head.eval()

    def post_step(self) -> int:
        # Do not increment global steps during evaluation; just return current value
        return int(self.shared_state.get("steps", 0))

    def run(self):
        env = make_env(self.env_name)
        loop = EnvironmentLoop(
            environment=env,
            actor=self,
            shared_state=self.shared_state,
        )
        # Ensure local modules are in sync before each episode.
        self.setup_local_modules()
        # Run a single episode (EnvironmentLoop will log via wandb using actor.label).
        loop.run(num_episodes=1)
        env.close()


class DmControlGymLikeWrapper:
    """Adapter to present a minimal gym-like API around a dm_control environment.

    It implements:
      - reset() -> (observation, info)
      - step(action) -> (observation, reward, terminated, truncated, info)
      - close()

    This keeps the rest of the code unchanged which expects gym-style tuples.
    """

    def __init__(self, env: dm_env.Environment):
        self._env = env

    def reset(self):
        ts = self._env.reset()
        return ts.observation, {}

    def step(self, action):
        # dm_control accepts numpy arrays in the action spec range.
        ts = self._env.step(action)
        obs = ts.observation
        reward = float(ts.reward or 0.0)
        terminated = bool(ts.last())
        truncated = False
        return obs, reward, terminated, truncated, {}

    def close(self):
        self._env.close()


def make_env(env_name: str, render_mode: str | None = None):
    domain, task = env_name.split("::")
    env = suite.load(domain, task)
    return DmControlGymLikeWrapper(env)


def flatten_observation(obs):
    if isinstance(obs, dict):
        parts = []
        for v in obs.values():
            parts.append(np.asarray(v).ravel())
        return np.concatenate(parts).astype(np.float32)
    else:
        return np.asarray(obs).ravel().astype(np.float32)


def run_policy_modules(
    obs_encoder: nn.Module,
    policy_head: nn.Module,
    obs: np.ndarray,
    device: torch.device,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    stochastic: bool,
) -> np.ndarray:
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        emb = obs_encoder(obs_t)
        mean, scale = policy_head(emb)
        if stochastic:
            action = mean + scale * torch.randn_like(mean)
        else:
            action = mean
        action = torch.clamp(action.squeeze(0), action_low, action_high)
    return action.cpu().numpy()


def train(
    env_name: str,
    max_actor_steps: int,
    batch_size: int,
    max_replay_size: int,
    min_replay_size: int,
    n_step: int,
    gamma: float,
    num_samples: int,
    target_policy_update_period: int,
    target_critic_update_period: int,
    lr: float,
    lr_dual: float,
    seed: int,
    log_dir: str,
    policy_sync_interval: int,
    clip_norm: float = 40.0,
    eval_interval_steps: int = 5000,
    max_learn_steps_per_call: int = 10,
    observations_per_iteration: int = 10000,
    learner_steps_per_iteration: int = 1000,
    action_penalization: bool = True,
    per_dim: bool = True,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Use dm_control suite to infer observation/action shapes and bounds.
    domain, task = env_name.split("::")
    probe_env = suite.load(domain, task)
    obs_spec = probe_env.observation_spec()
    # observation spec is often a dict of ArraySpec(s)
    if isinstance(obs_spec, dict):
        obs_dim = sum(int(np.prod(sp.shape)) for sp in obs_spec.values())
    else:
        obs_dim = int(np.prod(obs_spec.shape))
    action_spec = probe_env.action_spec()
    action_dim = int(np.prod(action_spec.shape))
    action_low = np.array(action_spec.minimum, dtype=np.float32)
    action_high = np.array(action_spec.maximum, dtype=np.float32)
    probe_env.close()
    del probe_env

    agent = MPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        device=device,
        batch_size=batch_size,
        max_replay_size=max_replay_size,
        n_step=n_step,
        gamma=gamma,
        num_samples=num_samples,
        target_policy_update_period=target_policy_update_period,
        target_critic_update_period=target_critic_update_period,
        lr_policy=lr,
        lr_critic=lr,
        lr_dual=lr_dual,
        min_replay_size=min_replay_size,
        clip_norm=clip_norm,
        action_penalization=action_penalization,
        per_dim=per_dim,
    )

    stop_event = threading.Event()
    shared_state = {"steps": 0}

    actor_obj = Actor(
        agent=agent,
        env_name=env_name,
        device=device,
        stop_event=stop_event,
        shared_state=shared_state,
        policy_sync_interval=policy_sync_interval,
        eval_interval_steps=eval_interval_steps,
        max_learn_steps_per_call=max_learn_steps_per_call,
    )

    # 1) Fill buffer until it reaches min_replay_size (blocking collection).
    while len(agent.replay) < max(1, agent.min_replay_size) and not stop_event.is_set():
        needed = max(1, agent.min_replay_size) - len(agent.replay)
        # collect in chunks to avoid overshooting excessively
        chunk = min(needed, observations_per_iteration)
        collected = actor_obj.collect_steps(chunk)

    # 2) Iterative loop: collect data, run learner, then evaluator.
    total_target = max_actor_steps
    while shared_state.get("steps", 0) < total_target and not stop_event.is_set():
        # collect specified number of observations (no learning during this collect)
        collected = actor_obj.collect_steps(observations_per_iteration)

        # run learner for up to learner_steps_per_iteration or until replay not ready
        learner_steps = 0
        while learner_steps < learner_steps_per_iteration and not stop_event.is_set():
            stats = agent.learn_step()
            if stats is None:
                # if learner ran out of data/allowed samples, break early
                break
            # log learner stats to wandb
            log_dict = {}
            for key, value in stats.items():
                if isinstance(value, torch.Tensor):
                    log_dict["learner/" + key] = float(value.detach().cpu().item())
                else:
                    log_dict["learner/" + key] = value
            log_dict["global_actor_step"] = int(shared_state.get("steps", 0))
            wandb.log(log_dict)
            learner_steps += 1

        actor_obj.evaluator.run()

    # finalize: save checkpoint as before
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{int(time.time())}.pt")
    replay_state = agent.replay.state_dict()
    checkpoint = {
        "obs_encoder": agent.obs_encoder.state_dict(),
        "policy_head": agent.policy_head.state_dict(),
        "critic": agent.critic.state_dict(),
        "target_obs_encoder": agent.target_obs_encoder.state_dict(),
        "target_policy_head": agent.target_policy_head.state_dict(),
        "target_critic": agent.target_critic.state_dict(),
        "mpo_loss": agent.mpo_loss.state_dict(),
        "_critic_optimizer": agent._critic_optimizer.state_dict(),
        "_policy_optimizer": agent._policy_optimizer.state_dict(),
        "_dual_optimizer": agent._dual_optimizer.state_dict(),
        # --- NEW: save value head and optimizer state ---
        "value_head": agent.value_head.state_dict(),
        "target_value_head": agent.target_value_head.state_dict(),
        "_value_optimizer": agent._value_optimizer.state_dict(),
        "learn_steps": agent._learn_steps,
        "seed": seed,
        "replay_state": replay_state,
    }
    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="cartpole::balance")
    parser.add_argument("--max_actor_steps", type=int, default=3_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--min_replay_size", type=int, default=512)
    parser.add_argument("--max_replay_size", type=int, default=1_000_000)
    parser.add_argument("--n_step", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--target_policy_update_period", type=int, default=25)
    parser.add_argument("--target_critic_update_period", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_dual", type=float, default=1e-2)
    parser.add_argument("--wandb_project", type=str, default="mpo_project")
    parser.add_argument("--wandb_entity", type=str, default="adrian-research")
    parser.add_argument("--wandb_group_prefix", type=str, default=None)
    parser.add_argument("--base_log_dir", type=str, default="./logs/mpo_experiment")
    parser.add_argument("--policy_sync_interval", type=int, default=1000)
    parser.add_argument(
        "--clip_norm",
        type=float,
        default=40.0,
        help="Gradient clipping max-norm; set to 0 to disable clipping.",
    )
    parser.add_argument(
        "--eval_interval_steps",
        type=int,
        default=5000,
        help="Run evaluation every N environment steps.",
    )
    parser.add_argument(
        "--max_learn_steps_per_call",
        type=int,
        default=10,
        help="Max learner steps executed each time maybe_learn is invoked.",
    )
    parser.add_argument(
        "--observations_per_iteration",
        type=int,
        default=10000,
        help="Number of environment steps to collect into replay each iteration.",
    )
    parser.add_argument(
        "--learner_steps_per_iteration",
        type=int,
        default=1000,
        help="Maximum number of learner steps to run each iteration.",
    )
    parser.add_argument(
        "--action_penalization",
        type=lambda s: s.lower() in ("true", "1", "t", "yes"),
        default=True,
        help="Enable action penalization (MO-MPO penalty). Accepts true/false.",
    )
    parser.add_argument(
        "--per_dim_constraining",
        type=lambda s: s.lower() in ("true", "1", "t", "yes"),
        default=True,
        help="Use per-dimension KL constraining. Accepts true/false.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed = int(time.time()) % 10000

    experiment_identifier = (
        args.env_name + f"__seed{seed}" + "_" + time.strftime("%Y%m%d-%H%M%S")
    )
    log_dir = os.path.join(
        args.base_log_dir, args.wandb_project + "_" + experiment_identifier
    )

    os.makedirs(log_dir, exist_ok=True)

    print("Experiment Configuration:")
    print(json.dumps(vars(args), indent=4))

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    wandb.init(
        name=experiment_identifier,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=(
            f"{args.wandb_group_prefix}_mpo_{args.env_name}"
            if args.wandb_group_prefix
            else f"mpo_{args.env_name}"
        ),
        config=vars(args),
        dir=log_dir,
    )

    train(
        env_name=args.env_name,
        max_actor_steps=args.max_actor_steps,
        min_replay_size=args.min_replay_size,
        batch_size=args.batch_size,
        max_replay_size=args.max_replay_size,
        n_step=args.n_step,
        gamma=args.gamma,
        num_samples=args.num_samples,
        target_policy_update_period=args.target_policy_update_period,
        target_critic_update_period=args.target_critic_update_period,
        lr=args.lr,
        lr_dual=args.lr_dual,
        seed=seed,
        log_dir=log_dir,
        policy_sync_interval=args.policy_sync_interval,
        clip_norm=args.clip_norm,
        eval_interval_steps=args.eval_interval_steps,
        max_learn_steps_per_call=args.max_learn_steps_per_call,
        # new params
        observations_per_iteration=args.observations_per_iteration,
        learner_steps_per_iteration=args.learner_steps_per_iteration,
        action_penalization=args.action_penalization,
        per_dim_constraining=args.per_dim_constraining,
    )

    wandb.finish()
