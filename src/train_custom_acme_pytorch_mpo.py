import argparse
import collections
import copy
import math
import random
import time
from typing import Tuple, cast, Dict, Union
import os
import json
import threading

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import shimmy
import torch.distributions as dist

from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict


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
    PyTorch version of compute_weights_and_temperature_loss.

    Returns:
        normalized_weights: [N, B]
        loss_temperature: scalar
    """

    # Temper Q-values; no gradient flows back into Q for the E-step dual update
    tempered_q_values = q_values.detach() / temperature

    # Softmax along action-sample dimension N
    normalized_weights = torch.softmax(tempered_q_values, dim=0).detach()

    # Dual loss for temperature (E-step Lagrange multiplier)
    # logsumexp over actions, averaged over batch
    q_logsumexp = torch.logsumexp(tempered_q_values, dim=0)  # [B]
    log_num_actions = math.log(q_values.shape[0])

    loss_temperature_inner = epsilon + q_logsumexp.mean() - log_num_actions  # scalar
    loss_temperature = temperature * loss_temperature_inner

    return normalized_weights, loss_temperature


def compute_nonparametric_kl_from_normalized_weights(
    normalized_weights: torch.Tensor,  # [N, B]
) -> torch.Tensor:
    """
    Estimate KL between non-parametric policy and target policy, from normalized weights.

    Returns:
        kl_nonparametric: [B]
    """
    num_action_samples = float(normalized_weights.shape[0])
    integrand = torch.log(
        num_action_samples * normalized_weights + _MPO_FLOAT_EPSILON
    )  # [N, B]
    return (normalized_weights * integrand).sum(dim=0)  # [B]


def compute_cross_entropy_loss(
    sampled_actions: torch.Tensor,  # [N, B, D]
    normalized_weights: torch.Tensor,  # [N, B]
    online_action_distribution: dist.Distribution,  # Independent(Normal)
) -> torch.Tensor:
    """
    PyTorch version of compute_cross_entropy_loss.

    Returns:
        scalar: mean over batch of the weighted negative log-prob.
    """
    # log_prob: [N, B]
    log_prob = online_action_distribution.log_prob(sampled_actions)

    # Weighted sum over actions (N), then mean over batch (B)
    loss_policy_gradient = -(log_prob * normalized_weights).sum(dim=0)  # [B]
    return loss_policy_gradient.mean()  # scalar


def compute_parametric_kl_penalty_and_dual_loss(
    kl: torch.Tensor,  # [B, D] or [B]
    alpha: torch.Tensor,  # [D] or [1], nn.Parameter (after softplus)
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch version of compute_parametric_kl_penalty_and_dual_loss.

    Returns:
        loss_kl: scalar (alpha-weighted KL regularizer)
        loss_alpha: scalar (dual loss for alpha)
    """
    # Mean over batch
    mean_kl = kl.mean(dim=0)  # [D] or [1]

    # KL regularization uses stop_gradient(alpha)
    loss_kl = (alpha.detach() * mean_kl).sum()

    # Dual loss updates alpha but does not backprop through mean_kl
    loss_alpha = (alpha * (epsilon - mean_kl.detach())).sum()

    return loss_kl, loss_alpha


_MPO_FLOAT_EPSILON = 1e-8


class MPOLoss(nn.Module):
    """
    PyTorch translation of the Sonnet MPOLoss with decoupled KL constraints.

    This expects Gaussian policies represented as Independent(Normal).
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
        """
        Args:
            epsilon: KL constraint for non-parametric policy (temperature).
            epsilon_mean: KL constraint for mean.
            epsilon_stddev: KL constraint for stddev.
            init_log_temperature: initial log-temperature (softplus parametrization).
            init_log_alpha_mean: initial log-alpha for mean constraint.
            init_log_alpha_stddev: initial log-alpha for stddev constraint.
            action_dim: dimensionality of action space D.
            per_dim_constraining: if True, constrain KL per dimension, else overall.
            action_penalization: use MO-MPO penalty for |a| > 1.
            epsilon_penalty: KL constraint for action penalty.
            dtype, device: standard Torch options.
        """
        super().__init__()

        self._epsilon = float(epsilon)
        self._epsilon_mean = float(epsilon_mean)
        self._epsilon_stddev = float(epsilon_stddev)
        self._epsilon_penalty = float(epsilon_penalty)

        self._per_dim_constraining = per_dim_constraining
        self._action_penalization = action_penalization

        # Register epsilons as buffers so they move with .to(...)
        self.register_buffer(
            "epsilon", torch.tensor(self._epsilon, dtype=dtype, device=device)
        )
        self.register_buffer(
            "epsilon_mean", torch.tensor(self._epsilon_mean, dtype=dtype, device=device)
        )
        self.register_buffer(
            "epsilon_stddev",
            torch.tensor(self._epsilon_stddev, dtype=dtype, device=device),
        )
        if self._action_penalization:
            self.register_buffer(
                "epsilon_penalty",
                torch.tensor(self._epsilon_penalty, dtype=dtype, device=device),
            )

        # Dual variables in log space
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
            self.log_penalty_temperature = nn.Parameter(
                torch.tensor([init_log_temperature], dtype=dtype, device=device)
            )

        self.min_log_temperature = torch.tensor(-18.0, dtype=dtype, device=device)
        self.min_log_alpha = torch.tensor(-18.0, dtype=dtype, device=device)

    def forward(
        self,
        actions: torch.Tensor,  # [N, B, D]
        q_values: torch.Tensor,  # [N, B]
        online_mean: torch.Tensor,
        online_scale: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the MPO loss and diagnostics.

        Args:
            online_action_distribution: online policy; batch [B], event [D].
            target_action_distribution: target policy; same shapes.
            actions: actions sampled from target policy; [N, B, D].
            q_values: Q(s, a); [N, B].

        Returns:
            loss: scalar MPO loss.
            stats: dict of scalars / small tensors for logging.
        """

        dtype = q_values.dtype

        # Clamp dual variables in log-space for stability
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
                self.log_penalty_temperature.data = torch.maximum(
                    self.log_penalty_temperature.data,
                    self.min_log_temperature.to(dtype),
                )

        # Softplus (instead of exp) for numerical stability
        temperature = F.softplus(self.log_temperature) + _MPO_FLOAT_EPSILON  # [1]
        alpha_mean = F.softplus(self.log_alpha_mean) + _MPO_FLOAT_EPSILON  # [D] or [1]
        alpha_stddev = F.softplus(self.log_alpha_stddev) + _MPO_FLOAT_EPSILON

        # Compute normalized weights for non-parametric E-step & temperature dual loss
        normalized_weights, loss_temperature = compute_weights_and_temperature_loss(
            q_values=q_values,  # [N, B]
            epsilon=self._epsilon,
            temperature=temperature,
        )

        # Non-parametric KL diagnostic
        kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
            normalized_weights
        )  # [B]

        # Optional MO-MPO action penalization
        if self._action_penalization:
            penalty_temperature = (
                F.softplus(self.log_penalty_temperature) + _MPO_FLOAT_EPSILON
            )

            # Cost: 0 inside [-1,1], negative quadratic outside (we *maximize* cost)
            diff_out_of_bound = actions - actions.clamp(-1.0, 1.0)  # [N, B, D]
            cost_out_of_bound = -diff_out_of_bound.norm(dim=-1)  # [N, B]

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

            # Combine weights and temperature losses
            normalized_weights = normalized_weights + penalty_normalized_weights
            loss_temperature = loss_temperature + loss_penalty_temperature
        else:
            penalty_kl_nonparametric = None

        # Decompose online policy into fixed-stddev and fixed-mean components
        fixed_stddev_dist = dist.Independent(
            dist.Normal(loc=online_mean, scale=target_scale), 1
        )
        fixed_mean_dist = dist.Independent(
            dist.Normal(loc=target_mean, scale=online_scale), 1
        )

        # Decomposed cross-entropy policy losses (E-step → M-step)
        loss_policy_mean = compute_cross_entropy_loss(
            sampled_actions=actions,
            normalized_weights=normalized_weights,
            online_action_distribution=fixed_stddev_dist,
        )
        loss_policy_stddev = compute_cross_entropy_loss(
            sampled_actions=actions,
            normalized_weights=normalized_weights,
            online_action_distribution=fixed_mean_dist,
        )

        # KL terms between target and decomposed policies
        per_dim = self._per_dim_constraining

        # KL for mean-constrained component
        kl_mean = _diag_normal_kl(
            p_mean=target_mean,
            p_std=target_scale,
            q_mean=fixed_stddev_dist.base_dist.loc,
            q_std=fixed_stddev_dist.base_dist.scale,
            per_dim=per_dim,
        )  # [B, D] or [B]

        # KL for stddev-constrained component
        kl_stddev = _diag_normal_kl(
            p_mean=target_mean,
            p_std=target_scale,
            q_mean=fixed_mean_dist.base_dist.loc,
            q_std=fixed_mean_dist.base_dist.scale,
            per_dim=per_dim,
        )  # [B, D] or [B]

        # Parametric KL penalization + dual losses
        loss_kl_mean, loss_alpha_mean = compute_parametric_kl_penalty_and_dual_loss(
            kl=kl_mean,
            alpha=alpha_mean,
            epsilon=self._epsilon_mean,
        )
        loss_kl_stddev, loss_alpha_stddev = compute_parametric_kl_penalty_and_dual_loss(
            kl=kl_stddev,
            alpha=alpha_stddev,
            epsilon=self._epsilon_stddev,
        )

        loss_policy = loss_policy_mean + loss_policy_stddev
        loss_kl_penalty = loss_kl_mean + loss_kl_stddev
        loss_dual = loss_alpha_mean + loss_alpha_stddev + loss_temperature

        loss = loss_policy + loss_kl_penalty + loss_dual

        # Diagnostics
        stats: Dict[str, torch.Tensor] = {}

        # Duals
        stats["train/dual_alpha_mean"] = alpha_mean.mean()
        stats["train/dual_alpha_stddev"] = alpha_stddev.mean()
        stats["train/dual_temperature"] = temperature.mean()

        # Loss terms
        stats["train/loss_policy"] = loss_policy.detach()
        stats["train/total_loss"] = loss.detach()
        stats["train/loss_alpha"] = (loss_alpha_mean + loss_alpha_stddev).detach()
        stats["train/loss_temperature"] = loss_temperature.detach()

        # KL diagnostics
        stats["train/kl_q_rel"] = kl_nonparametric.mean() / self._epsilon

        if self._action_penalization and penalty_kl_nonparametric is not None:
            stats["train/penalty_kl_q_rel"] = (
                penalty_kl_nonparametric.mean() / self._epsilon_penalty
            )

        stats["train/kl_mean_rel"] = kl_mean.mean() / self._epsilon_mean
        stats["train/kl_stddev_rel"] = kl_stddev.mean() / self._epsilon_stddev
        if self._per_dim_constraining:
            # per-dim summaries
            kl_mean_batch_mean = kl_mean.mean(dim=0)  # [D]
            kl_stddev_batch_mean = kl_stddev.mean(dim=0)  # [D]

            stats["train/kl_mean_rel_min"] = (
                kl_mean_batch_mean.min() / self._epsilon_mean
            )
            stats["train/kl_mean_rel_max"] = (
                kl_mean_batch_mean.max() / self._epsilon_mean
            )
            stats["train/kl_stddev_rel_min"] = (
                kl_stddev_batch_mean.min() / self._epsilon_stddev
            )
            stats["train/kl_stddev_rel_max"] = (
                kl_stddev_batch_mean.max() / self._epsilon_stddev
            )

        # Q statistics
        stats["train/q_min"] = q_values.min(dim=0).values.mean()
        stats["train/q_max"] = q_values.max(dim=0).values.mean()

        # Policy stddev statistics for the online policy
        pi_stddev = online_scale  # [B, D]
        stats["train/pi_stddev_min"] = pi_stddev.min(dim=-1).values.mean()
        stats["train/pi_stddev_max"] = pi_stddev.max(dim=-1).values.mean()
        stats["train/pi_stddev_cond"] = (
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
        clipping=True,
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
        self.clipping = clipping

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
            init_log_alpha_stddev=100.0,
        ).to(device)

        self._critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self._policy_optimizer = optim.Adam(self.policy_head.parameters(), lr=lr_policy)
        self._dual_optimizer = optim.Adam(self.mpo_loss.parameters(), lr=lr_dual)

        self._replay_lock = threading.Lock()
        self._param_lock = threading.Lock()

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
        with self._param_lock:
            obs_encoder.load_state_dict(self.obs_encoder.state_dict())
            policy_head.load_state_dict(self.policy_head.state_dict())

    def select_action(self, obs: np.ndarray, stochastic: bool = True) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with self._param_lock:
            with torch.no_grad():
                emb = self.obs_encoder(obs_t)
                mean, scale = self.policy_head(emb)
                if stochastic:
                    action = mean + scale * torch.randn_like(mean)
                else:
                    action = mean
                action = torch.clamp(
                    action.squeeze(0), self.action_low, self.action_high
                )
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
            with self._replay_lock:
                self.replay.add(data)
                self._num_inserts += 1

    def _can_sample(self):
        with self._replay_lock:
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

        with self._replay_lock:
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

        # critic loss (TD)
        target = rewards_b + discounts_b * q_t

        q_tm1 = self.critic(obs_b, actions_b)

        # Match Acme/td_learning: TD error = target - v_tm1, loss = 0.5 * td_error^2
        td_error = target - q_tm1
        critic_loss = 0.5 * (td_error**2).mean()
        # small diagnostics for logging
        td_error_mean = float(td_error.abs().mean().item())
        target_mean = float(target.mean().item())

        with torch.no_grad():
            emb_o_t = self.target_obs_encoder(next_obs_b)
        online_mean, online_scale = self.policy_head(emb_o_t)

        # target mean/scale already computed as t_mean/t_scale (from above, no_grad)
        # compute MPO loss with sampled_actions and q_samples
        total_loss, stats = self.mpo_loss(
            sampled_actions, q_samples, online_mean, online_scale, t_mean, t_scale
        )

        with self._param_lock:
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.clipping:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 40.0)
            self._critic_optimizer.step()

            self._policy_optimizer.zero_grad()
            self._dual_optimizer.zero_grad()
            total_loss.backward()
            if self.clipping:
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_head.parameters()),
                    40.0,
                )
            self._policy_optimizer.step()
            self._dual_optimizer.step()

        self._learn_steps += 1

        # periodic target hard updates
        # Update target policy head (periodic).
        if self._learn_steps % self.target_policy_update_period == 0:
            self.target_policy_head.load_state_dict(self.policy_head.state_dict())
            self.target_obs_encoder.load_state_dict(self.obs_encoder.state_dict())
        # Update target critic and the target observation encoder (periodic).
        if self._learn_steps % self.target_critic_update_period == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())

        fetches = {
            "train/critic_loss": float(critic_loss.item()),
            "train/learn_steps": self._learn_steps,
        }
        fetches.update(
            {
                "train/td_error_mean": td_error_mean,
                "train/td_target_mean": target_mean,
            }
        )
        fetches.update(stats)

        if self._learn_steps % 100 == 0:
            print(
                f"[Learner step {self._learn_steps}] "
                f"critic_loss={critic_loss.item():.4f} "
                f"td_error_mean={td_error_mean:.4f} "
                f"td_target_mean={target_mean:.4f}"
            )
        return fetches


def make_env(env_name: str, render_mode: str | None = None):
    domain, task = env_name.split("::")
    return gym.make(f"dm_control/{domain}-{task}-v0", render_mode=render_mode)


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
    max_eval_actor_steps: int,
    eval_freq: int,
    policy_sync_interval: int,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    probe_env = make_env(env_name)
    obs0, _ = probe_env.reset()
    obs_space = probe_env.observation_space
    if isinstance(obs_space, gym.spaces.Dict):
        obs_dim = sum(int(np.prod(sp.shape)) for sp in obs_space.spaces.values())
    else:
        obs_dim = int(np.prod(obs_space.shape))
    act_space = probe_env.action_space
    action_dim = int(np.prod(act_space.shape))
    action_low = act_space.low
    action_high = act_space.high
    probe_env.close()

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
    )

    stop_event = threading.Event()
    step_lock = threading.Lock()
    shared_state = {"steps": 0}
    policy_sync_interval = max(1, policy_sync_interval)

    def actor_loop():
        env = make_env(env_name)
        try:
            obs, _ = env.reset()
            obs_flat_local = flatten_observation(obs)
            actor_encoder = copy.deepcopy(agent.obs_encoder).to(device)
            actor_policy_head = copy.deepcopy(agent.policy_head).to(device)
            agent.copy_policy_to(actor_encoder, actor_policy_head)
            actor_encoder.eval()
            actor_policy_head.eval()
            episode_return = 0.0
            episode_len = 0
            sync_counter = policy_sync_interval
            while not stop_event.is_set():
                if sync_counter >= policy_sync_interval:
                    agent.copy_policy_to(actor_encoder, actor_policy_head)
                    actor_encoder.eval()
                    actor_policy_head.eval()
                    sync_counter = 0

                action = run_policy_modules(
                    actor_encoder,
                    actor_policy_head,
                    obs_flat_local,
                    device,
                    agent.action_low,
                    agent.action_high,
                    stochastic=True,
                )
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_flat = flatten_observation(next_obs)
                step_discount = 0.0 if done else 1.0
                agent.store_transition(
                    obs_flat_local,
                    action,
                    float(reward),
                    step_discount,
                    next_flat,
                    done,
                )
                episode_return += float(reward)
                episode_len += 1
                obs_flat_local = next_flat
                sync_counter += 1

                with step_lock:
                    shared_state["steps"] += 1
                    global_step = shared_state["steps"]

                if done:
                    wandb.log(
                        {
                            "train/episode_return": episode_return,
                            "train/episode_length": episode_len,
                        },
                    )
                    print(
                        f"[Actor step {global_step}] return={episode_return:.2f} len={episode_len}"
                    )
                    obs, _ = env.reset()
                    obs_flat_local = flatten_observation(obs)
                    episode_return = 0.0
                    episode_len = 0

                if global_step >= max_actor_steps:
                    stop_event.set()
                    break
        except Exception as exc:
            stop_event.set()
            print(f"[Actor] error: {exc}")
        finally:
            env.close()

    def learner_loop():
        try:
            while not stop_event.is_set():
                stats = agent.learn_step()
                if stats is None:
                    time.sleep(0.01)
                    continue
                with step_lock:
                    global_step = shared_state["steps"]
                log_dict: Dict[str, Union[float, int]] = {}
                for key, value in stats.items():
                    if isinstance(value, torch.Tensor):
                        log_dict[key] = float(value.detach().cpu().item())
                    else:
                        log_dict[key] = value
                log_dict["train/global_actor_step"] = global_step
                wandb.log(log_dict)
        except Exception as exc:
            stop_event.set()
            print(f"[Learner] error: {exc}")

    def evaluator_loop():
        env = make_env(env_name)
        try:
            eval_encoder = copy.deepcopy(agent.obs_encoder).to(device)
            eval_policy_head = copy.deepcopy(agent.policy_head).to(device)
            agent.copy_policy_to(eval_encoder, eval_policy_head)
            eval_encoder.eval()
            eval_policy_head.eval()
            next_eval_step = eval_freq
            while not stop_event.is_set():
                if eval_freq <= 0 or max_eval_actor_steps <= 0:
                    break
                with step_lock:
                    global_step = shared_state["steps"]
                if global_step >= next_eval_step and global_step > 0:
                    agent.copy_policy_to(eval_encoder, eval_policy_head)
                    eval_encoder.eval()
                    eval_policy_head.eval()
                    eval_returns = []
                    eval_lengths = []
                    eval_steps = 0
                    while eval_steps < max_eval_actor_steps and not stop_event.is_set():
                        obs, _ = env.reset()
                        obs_flat_local = flatten_observation(obs)
                        done = False
                        ep_ret = 0.0
                        ep_len = 0
                        while (
                            not done
                            and eval_steps < max_eval_actor_steps
                            and not stop_event.is_set()
                        ):
                            action = run_policy_modules(
                                eval_encoder,
                                eval_policy_head,
                                obs_flat_local,
                                device,
                                agent.action_low,
                                agent.action_high,
                                stochastic=False,
                            )
                            next_obs, reward, terminated, truncated, _ = env.step(
                                action
                            )
                            done = terminated or truncated
                            obs_flat_local = flatten_observation(next_obs)
                            ep_ret += float(reward)
                            ep_len += 1
                            eval_steps += 1
                        eval_returns.append(ep_ret)
                        eval_lengths.append(ep_len)
                    if eval_returns:
                        mean_r = float(np.mean(eval_returns))
                        std_r = float(np.std(eval_returns))
                        mean_len = float(np.mean(eval_lengths))
                        wandb.log(
                            {
                                "eval/mean_reward": mean_r,
                                "eval/std_return": std_r,
                                "eval/mean_ep_length": mean_len,
                            },
                        )
                        print(
                            f"[Eval @ step {global_step}] mean_reward={mean_r:.2f} std={std_r:.2f} mean_len={mean_len:.1f}"
                        )
                    next_eval_step += eval_freq
                else:
                    time.sleep(0.5)
        except Exception as exc:
            stop_event.set()
            print(f"[Evaluator] error: {exc}")
        finally:
            env.close()

    threads = [
        threading.Thread(target=actor_loop, name="actor_thread"),
        threading.Thread(target=learner_loop, name="learner_thread"),
    ]
    if eval_freq > 0 and max_eval_actor_steps > 0:
        threads.append(threading.Thread(target=evaluator_loop, name="evaluator_thread"))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stop_event.set()
    step = shared_state["steps"]

    try:
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{int(time.time())}.pt")
        with agent._replay_lock:
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
            "learn_steps": agent._learn_steps,
            "step": step,
            "seed": seed,
            "replay_state": replay_state,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_names",
        type=str,
        default="cartpole::balance",
        help="Comma-separated list of environment names to train on",
    )
    parser.add_argument("--env_iterations", type=int, default=1)
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
    parser.add_argument("--lr_dual", type=float, default=5e-4)
    parser.add_argument("--wandb_project", type=str, default="mpo_project")
    parser.add_argument("--wandb_entity", type=str, default="adrian-research")
    parser.add_argument("--wandb_group_prefix", type=str, default=None)
    parser.add_argument("--base_log_dir", type=str, default="./logs/mpo_experiment")
    parser.add_argument("--eval_freq", type=int, default=3000)
    parser.add_argument("--max_eval_actor_steps", type=int, default=3000)
    parser.add_argument("--policy_sync_interval", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env_names = args.env_names.split(",")
    for env_name in env_names:
        for iteration in range(args.env_iterations):
            print(
                f"Training on environment: {env_name}. Starting iteration {iteration + 1}/{args.env_iterations}"
            )
            start_time = time.time()

            seed = int(time.time()) % 10000 + iteration * 1000

            experiment_identifier = (
                env_name
                + "__iter"
                + str(iteration + 1)
                + f"_seed{seed}"
                + "_"
                + time.strftime("%Y%m%d-%H%M%S")
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
                    f"{args.wandb_group_prefix}_mpo_{env_name}"
                    if args.wandb_group_prefix
                    else f"mpo_{env_name}"
                ),
                config=vars(args),
                dir=log_dir,
            )

            train(
                env_name=env_name,
                max_actor_steps=args.max_actor_steps,
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
                eval_freq=args.eval_freq,
                max_eval_actor_steps=args.max_eval_actor_steps,
                min_replay_size=args.min_replay_size,
                log_dir=log_dir,
                policy_sync_interval=args.policy_sync_interval,
            )

            wandb.finish()
            end_time = time.time()
            print(
                f"Iteration {iteration + 1}/{args.env_iterations} completed in "
                f"{end_time - start_time:.2f} seconds."
            )
