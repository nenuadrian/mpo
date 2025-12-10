import argparse
import collections
import copy
import math
import random
import time
from typing import Deque, Tuple
import os
import json

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb


class NStepTransition:
    __slots__ = ("obs", "action", "reward", "discount", "next_obs", "done")

    def __init__(self, obs, action, reward, discount, next_obs, done):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.discount = discount
        self.next_obs = next_obs
        self.done = done


class NStepReplay:
    """Ring buffer storing N-step transitions assembled from an on-policy buffer."""

    def __init__(self, max_size: int, n_step: int, gamma: float):
        self._max_size = max_size
        self._buffer: Deque[NStepTransition] = collections.deque(maxlen=max_size)
        self._n_step = n_step
        self._gamma = gamma
        self._traj: Deque[Tuple] = collections.deque()

    def append(self, obs, action, reward, discount, next_obs, done):
        self._traj.append((obs, action, reward, discount, next_obs, done))
        # create N-step transitions when enough collected or on terminal
        while len(self._traj) >= self._n_step or (self._traj and self._traj[0][5]):
            R = 0.0
            cur_discount = 1.0
            next_o = None
            terminal = False
            for i, (_, _, r, d, nobs, dn) in enumerate(self._traj):
                R += cur_discount * r
                cur_discount *= self._gamma * d
                next_o = nobs
                terminal = dn
                if i == self._n_step - 1:
                    break
            obs0, a0, _, _, _, _ = self._traj[0]
            self._buffer.append(
                NStepTransition(obs0, a0, R, cur_discount, next_o, terminal)
            )
            self._traj.popleft()
            if not self._traj:
                break

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self._buffer), size=batch_size)
        batch = [self._buffer[i] for i in idx]
        obs = np.stack([b.obs for b in batch], axis=0)
        actions = np.stack([b.action for b in batch], axis=0)
        rewards = np.stack([b.reward for b in batch], axis=0)
        discounts = np.stack([b.discount for b in batch], axis=0)
        next_obs = np.stack([b.next_obs for b in batch], axis=0)
        dones = np.stack([b.done for b in batch], axis=0)
        return obs, actions, rewards, discounts, next_obs, dones

    def __len__(self):
        return len(self._buffer)


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


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
        init_scale: float = 0.3,
        min_scale: float = 1e-6,
        tanh_mean: bool = False,
        fixed_scale: bool = False,
    ):
        super().__init__()
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.fixed_scale = fixed_scale
        self.tanh_mean = tanh_mean
        self.init_scale = init_scale
        self.min_scale = min_scale
        if not fixed_scale:
            # output positive scale via softplus of a linear layer
            self.log_scale_layer = nn.Linear(input_dim, action_dim)
        else:
            self.register_buffer("_fixed_scale", torch.tensor(init_scale))
        self.apply(init_weights)

        # Make log-scale start near zero so softplus(0) leads to init_scale after rescaling.
        if not fixed_scale:
            with torch.no_grad():
                # zero weights/bias for predictable initial stddev = init_scale
                self.log_scale_layer.weight.zero_()
                if self.log_scale_layer.bias is not None:
                    self.log_scale_layer.bias.zero_()
        # make mean bias zero for stable initial means
        with torch.no_grad():
            if self.mean_layer.bias is not None:
                self.mean_layer.bias.zero_()

    def forward(self, x):
        mean = self.mean_layer(x)
        if self.tanh_mean:
            mean = torch.tanh(mean)
        if self.fixed_scale:
            scale = torch.ones_like(mean) * float(self._fixed_scale)
        else:
            log_scale = self.log_scale_layer(x)
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
            final = self.net[-1]
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


class MPOLoss(nn.Module):
    """
    MPO loss with dual variables:
      - temperature (scalar)
      - alpha_mean (per-dim or scalar)
      - alpha_stddev (per-dim or scalar)
    Includes optional MO-MPO action penalization.
    """

    def __init__(
        self,
        action_dim: int,
        per_dim_constraining: bool = True,
        action_penalization: bool = True,
        epsilon: float = 1e-1,
        epsilon_mean: float = 2.5e-3,
        epsilon_stddev: float = 1e-6,
        epsilon_penalty: float = 1e-3,
        init_log_temperature: float = 10.0,
        init_log_alpha_mean: float = 10.0,
        init_log_alpha_stddev: float = 1000.0,
        min_log: float = -18.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.per_dim = per_dim_constraining
        self.action_penalization = action_penalization
        self._epsilon = epsilon
        self._epsilon_mean = epsilon_mean
        self._epsilon_stddev = epsilon_stddev
        self._epsilon_penalty = epsilon_penalty
        self._min_log = min_log

        # dual variables (log-space)
        self.log_temperature = nn.Parameter(
            torch.tensor([init_log_temperature], dtype=torch.float32)
        )
        alpha_shape = (action_dim,) if self.per_dim else (1,)
        self.log_alpha_mean = nn.Parameter(
            torch.full(alpha_shape, init_log_alpha_mean, dtype=torch.float32)
        )
        self.log_alpha_stddev = nn.Parameter(
            torch.full(alpha_shape, init_log_alpha_stddev, dtype=torch.float32)
        )
        if self.action_penalization:
            self.log_penalty_temperature = nn.Parameter(
                torch.tensor([init_log_temperature], dtype=torch.float32)
            )

    def forward(
        self,
        online_mean: torch.Tensor,
        online_scale: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
        sampled_actions: torch.Tensor,  # [N,B,D]
        q_values: torch.Tensor,  # [N,B]
    ):
        # ensure positivity with softplus, clamp logs to avoid extreme negatives
        with torch.no_grad():
            self.log_temperature.data.clamp_(min=self._min_log)
            self.log_alpha_mean.data.clamp_(min=self._min_log)
            self.log_alpha_stddev.data.clamp_(min=self._min_log)
            if self.action_penalization:
                self.log_penalty_temperature.data.clamp_(min=self._min_log)

        temperature = F.softplus(self.log_temperature) + 1e-8  # scalar
        alpha_mean = F.softplus(self.log_alpha_mean) + 1e-8  # (D,) or (1,)
        alpha_std = F.softplus(self.log_alpha_stddev) + 1e-8

        N, B, D = sampled_actions.shape

        # E-step: tempered Q-values -> normalized weights
        tempered_q = q_values.detach() / temperature  # [N,B]
        normalized_weights = torch.softmax(tempered_q, dim=0).detach()  # [N,B]

        # temperature dual loss (matches TF math)
        q_logsumexp = torch.logsumexp(tempered_q, dim=0)  # [B]
        log_num_actions = math.log(float(N))
        loss_temperature = (
            self._epsilon + q_logsumexp.mean() - log_num_actions
        ) * temperature

        # optional action penalization (MO-MPO)
        penalty_kl_q = None
        if self.action_penalization:
            penalty_temperature = F.softplus(self.log_penalty_temperature) + 1e-8
            diff_out = sampled_actions - sampled_actions.clamp(-1.0, 1.0)
            cost = -torch.norm(diff_out, dim=-1)  # [N,B]
            penalty_tempered = cost.detach() / penalty_temperature
            penalty_w = torch.softmax(penalty_tempered, dim=0).detach()
            penalty_q_logsumexp = torch.logsumexp(penalty_tempered, dim=0)
            loss_penalty_temp = (
                self._epsilon_penalty + penalty_q_logsumexp.mean() - math.log(float(N))
            ) * penalty_temperature
            normalized_weights = normalized_weights + penalty_w
            loss_temperature = loss_temperature + loss_penalty_temp

        # Decompose online policy into fixed-std and fixed-mean distributions
        fixed_std_mean = online_mean  # for fixed std distribution (mean variable)
        fixed_std_scale = target_scale
        fixed_mean_mean = target_mean
        fixed_mean_scale = online_scale

        # Cross-entropy loss: - E_{nonparametric}[ log pi_online(sampled_action) ]
        def weighted_cross_entropy(means, scales):
            means_exp = means.unsqueeze(0).expand(N, -1, -1)  # [N,B,D]
            scales_exp = scales.unsqueeze(0).expand(N, -1, -1)  # [N,B,D]
            # log prob of Gaussian diagonal
            logp = -0.5 * (
                ((sampled_actions - means_exp) / scales_exp) ** 2
                + 2.0 * torch.log(scales_exp)
                + math.log(2 * math.pi)
            )
            logp = logp.sum(dim=-1)  # [N,B]
            loss = -(logp * normalized_weights).sum(dim=0)  # [B]
            return loss.mean()

        loss_policy_mean = weighted_cross_entropy(fixed_std_mean, fixed_std_scale)
        loss_policy_std = weighted_cross_entropy(fixed_mean_mean, fixed_mean_scale)

        # KL computations: KL(target || fixed) per-dim
        def kl_diag(m_p, s_p, m_q, s_q):
            # Correct KL(P || Q) for diagonal normals with stddevs s_p, s_q
            var_p = s_p**2
            var_q = s_q**2
            # KL = log(s_q/s_p) + (var_p + (m_p - m_q)^2) / (2 * var_q) - 0.5
            term = (
                torch.log(s_q / s_p) + (var_p + (m_p - m_q) ** 2) / (2.0 * var_q) - 0.5
            )
            return term  # [B,D]

        kl_mean = kl_diag(
            target_mean, target_scale, fixed_std_mean, fixed_std_scale
        )  # [B,D]
        kl_std = kl_diag(
            target_mean, target_scale, fixed_mean_mean, fixed_mean_scale
        )  # [B,D]

        if not self.per_dim:
            kl_mean = kl_mean.sum(dim=-1, keepdim=True)  # [B,1]
            kl_std = kl_std.sum(dim=-1, keepdim=True)

        mean_kl = kl_mean.mean(dim=0)  # (D,) or (1,)
        std_kl = kl_std.mean(dim=0)

        # alpha-weighted KL penalties and dual losses
        loss_kl_mean = (alpha_mean.detach() * mean_kl).sum()
        loss_kl_std = (alpha_std.detach() * std_kl).sum()

        loss_alpha_mean = (alpha_mean * (self._epsilon_mean - mean_kl.detach())).sum()
        loss_alpha_std = (alpha_std * (self._epsilon_stddev - std_kl.detach())).sum()

        loss_policy = loss_policy_mean + loss_policy_std
        loss_kl_penalty = loss_kl_mean + loss_kl_std
        loss_dual = loss_alpha_mean + loss_alpha_std + loss_temperature

        loss = loss_policy + loss_kl_penalty + loss_dual

        # Diagnostics: compute non-parametric KL and other stats (match TF)
        with torch.no_grad():
            # non-parametric KL estimate: KL(nonparam || target) per-batch
            eps = 1e-8
            num_actions = float(N)
            integrand = torch.log(num_actions * normalized_weights + eps)
            kl_nonparametric = (normalized_weights * integrand).sum(dim=0)  # [B]

            stats = {
                "train/dual_alpha_mean": float(alpha_mean.mean().item()),
                "train/dual_alpha_stddev": float(alpha_std.mean().item()),
                "train/dual_temperature": float(temperature.item()),
                "train/loss_policy": float(loss_policy.item()),
                "train/loss_alpha": float((loss_alpha_mean + loss_alpha_std).item()),
                "train/loss_temperature": float(loss_temperature.item()),
                "train/kl_q_rel": float(
                    kl_nonparametric.mean().item() / float(self._epsilon)
                ),
            }

            # Q stats
            stats["train/q_min"] = float(q_values.min(dim=0)[0].mean().item())
            stats["train/q_max"] = float(q_values.max(dim=0)[0].mean().item())

            # pi stddev stats: online_scale shape is [B,D]
            pi_stddev = online_scale.detach()
            pi_std_min = pi_stddev.min(dim=-1)[0]  # [B]
            pi_std_max = pi_stddev.max(dim=-1)[0]  # [B]
            stats["train/pi_stddev_min"] = float(pi_std_min.mean().item())
            stats["train/pi_stddev_max"] = float(pi_std_max.mean().item())
            # condition number: mean over batch of (max/min)
            cond = (pi_std_max / (pi_std_min + 1e-12)).mean().item()
            stats["train/pi_stddev_cond"] = float(cond)

            # KL mean/std relative
            stats["train/kl_mean_rel"] = float(
                mean_kl.mean().item() / float(self._epsilon_mean)
            )
            stats["train/kl_stddev_rel"] = float(
                std_kl.mean().item() / float(self._epsilon_stddev)
            )

            if self.per_dim:
                # per-dim min/max (match TF logging)
                kl_mean_per_dim = kl_mean.mean(dim=0)  # [D] or [1]
                kl_std_per_dim = kl_std.mean(dim=0)
                stats["train/kl_mean_rel_min"] = float(
                    kl_mean_per_dim.min().item() / float(self._epsilon_mean)
                )
                stats["train/kl_mean_rel_max"] = float(
                    kl_mean_per_dim.max().item() / float(self._epsilon_mean)
                )
                stats["train/kl_stddev_rel_min"] = float(
                    kl_std_per_dim.min().item() / float(self._epsilon_stddev)
                )
                stats["train/kl_stddev_rel_max"] = float(
                    kl_std_per_dim.max().item() / float(self._epsilon_stddev)
                )

            if self.action_penalization:
                # penalty KL relative — compute using penalty normalized weights if available
                # The TF version logs a penalty_kl_q_rel; we try to approximate: recompute penalty_kl if possible
                diff_out = sampled_actions - sampled_actions.clamp(-1.0, 1.0)
                cost = -torch.norm(diff_out, dim=-1)  # [N,B]
                penalty_temperature = F.softplus(self.log_penalty_temperature) + 1e-8
                penalty_tempered = cost.detach() / penalty_temperature
                penalty_w = torch.softmax(penalty_tempered, dim=0).detach()
                integrand_pen = torch.log(num_actions * penalty_w + eps)
                penalty_kl_nonparametric = (penalty_w * integrand_pen).sum(dim=0)
                stats["train/penalty_kl_q_rel"] = float(
                    penalty_kl_nonparametric.mean().item()
                    / float(self._epsilon_penalty)
                )

        return loss, stats


class MPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low,
        action_high,
        device: torch.device,
        policy_hidden=(256, 256, 256),
        critic_hidden=(512, 512, 256),
        init_scale=0.7,
        gamma=0.99,
        n_step=5,
        batch_size=256,
        replay_size=100000,
        num_samples=20,
        target_policy_update_period=25,
        target_critic_update_period=100,
        lr_policy=1e-4,
        lr_critic=1e-4,
        lr_dual=1e-2,
        clipping=True,
        action_penalization=True,
        per_dim=True,
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
        self.critic = CriticNetwork(critic_input_dim, layer_sizes=critic_hidden).to(
            device
        )

        # target networks (hard copy)
        self.target_obs_encoder = copy.deepcopy(self.obs_encoder).to(device)
        self.target_policy_head = copy.deepcopy(self.policy_head).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)

        # optimizers
        # Critic optimizer trains both critic and observation encoder (like TF learner).
        self.critic_opt = optim.Adam(
            list(self.obs_encoder.parameters()) + list(self.critic.parameters()),
            lr=lr_critic,
        )
        # Policy optimizer trains only the policy head (policy shouldn't train the obs encoder).
        self.policy_opt = optim.Adam(list(self.policy_head.parameters()), lr=lr_policy)

        # MPO loss with duals (parameters included)
        self.mpo_loss = MPOLoss(
            action_dim=action_dim,
            per_dim_constraining=per_dim,
            action_penalization=action_penalization,
        ).to(device)
        self.dual_opt = optim.Adam(self.mpo_loss.parameters(), lr=lr_dual)

        # replay
        self.replay = NStepReplay(max_size=replay_size, n_step=n_step, gamma=gamma)

        self._learn_steps = 0

    def select_action(self, obs: np.ndarray, stochastic: bool = True) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            emb = self.obs_encoder(obs_t)
            mean, scale = self.policy_head(emb)
            if stochastic:
                eps = torch.randn_like(mean)
                action = mean + scale * eps
            else:
                action = mean
            action = action.squeeze(0)
            # stable/clamped action output
            action = torch.clamp(action, self.action_low, self.action_high)
            return action.cpu().numpy()

    def store_transition(self, obs, action, reward, discount, next_obs, done):
        self.replay.append(obs, action, reward, discount, next_obs, done)

    def learn_step(self):
        if len(self.replay) < max(1000, self.batch_size):
            return None

        obs_b, actions_b, rewards_b, discounts_b, next_obs_b, dones_b = (
            self.replay.sample(self.batch_size)
        )

        device = self.device
        obs_b = torch.tensor(obs_b, dtype=torch.float32, device=device)
        actions_b = torch.tensor(actions_b, dtype=torch.float32, device=device)
        rewards_b = torch.tensor(rewards_b, dtype=torch.float32, device=device)
        discounts_b = torch.tensor(discounts_b, dtype=torch.float32, device=device)
        next_obs_b = torch.tensor(next_obs_b, dtype=torch.float32, device=device)

        # --- compute target Q via target policy sampling ---
        with torch.no_grad():
            # Use target observation encoder to get embeddings for next states.
            emb_next = self.target_obs_encoder(next_obs_b)
            t_mean, t_scale = self.target_policy_head(emb_next)  # (B,D)
            N = self.num_samples
            # sample N actions per batch element
            t_mean_exp = t_mean.unsqueeze(0).expand(N, -1, -1)  # (N,B,D)
            t_scale_exp = t_scale.unsqueeze(0).expand(N, -1, -1)
            eps = torch.randn_like(t_mean_exp)
            sampled_actions = t_mean_exp + t_scale_exp * eps  # (N,B,D)
            # clamp sampled target actions to action bounds to avoid extremely large Qs
            al = self.action_low.view(1, 1, -1)
            ah = self.action_high.view(1, 1, -1)
            sampled_actions = torch.clamp(sampled_actions, al, ah)
            # evaluate target critic on tiled embeddings + sampled actions
            tiled_emb_next = (
                emb_next.unsqueeze(0).expand(N, -1, -1).reshape(N * self.batch_size, -1)
            )
            sampled_actions_resh = sampled_actions.reshape(N * self.batch_size, -1)
            critic_inputs = torch.cat([tiled_emb_next, sampled_actions_resh], dim=-1)
            q_samples = self.target_critic(critic_inputs).view(
                N, self.batch_size
            )  # (N,B)
            # guard against NaN / Inf in target Qs
            q_samples = torch.nan_to_num(q_samples, nan=0.0, posinf=1e6, neginf=-1e6)
            q_t = q_samples.mean(dim=0)  # (B,)

        # critic loss (TD)
        target = rewards_b + discounts_b * q_t

        # clamp actions from replay batch before critic eval to keep inputs within action bounds
        actions_b = torch.clamp(
            actions_b, self.action_low.view(1, -1), self.action_high.view(1, -1)
        )
        # Use the observation encoder (trained by critic) to get embeddings for current obs.
        obs_emb = self.obs_encoder(obs_b)
        critic_inputs_tm1 = torch.cat([obs_emb, actions_b], dim=-1)
        q_tm1 = self.critic(critic_inputs_tm1)
        critic_loss = F.mse_loss(q_tm1, target)

        # policy update using MPO loss
        # Use detached target embeddings (o_t) for policy computation so the policy head
        # updates but not the observation encoder (mirrors TF stop_gradient on o_t).
        emb_o_t = emb_next.detach()
        online_mean, online_scale = self.policy_head(emb_o_t)
        # target mean/scale already computed as t_mean/t_scale (from above, no_grad)
        # compute MPO loss with sampled_actions and q_samples
        total_loss, stats = self.mpo_loss(
            online_mean, online_scale, t_mean, t_scale, sampled_actions, q_samples
        )

        # update critic
        self.critic_opt.zero_grad()
        critic_loss.backward()
        if self.clipping:
            torch.nn.utils.clip_grad_norm_(
                list(self.obs_encoder.parameters()) + list(self.critic.parameters()),
                40.0,
            )
        self.critic_opt.step()

        # update policy and duals together (total_loss contains dual terms)
        self.policy_opt.zero_grad()
        self.dual_opt.zero_grad()
        total_loss.backward()
        if self.clipping:
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_head.parameters()),
                40.0,
            )
        self.policy_opt.step()
        # dual optimizer step
        self.dual_opt.step()

        self._learn_steps += 1

        # periodic target hard updates
        # Update target policy head (periodic).
        if self._learn_steps % self.target_policy_update_period == 0:
            self.target_policy_head.load_state_dict(self.policy_head.state_dict())
        # Update target critic and the target observation encoder (periodic).
        if self._learn_steps % self.target_critic_update_period == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.target_obs_encoder.load_state_dict(self.obs_encoder.state_dict())

        fetches = {
            "train/critic_loss": float(critic_loss.item()),
            "train/learn_steps": self._learn_steps,
        }
        fetches.update(stats)
        return fetches


def make_env(env_name: str):
    # dm_control names are like "dm_control/domain-task-v0" or you pass "cartpole::balance"
    if env_name.startswith("dm_control/") or "::" in env_name:
        import shimmy

        if "::" in env_name:
            domain, task = env_name.split("::")
            return gym.make(f"dm_control/{domain}-{task}-v0")
        else:
            return gym.make(env_name)
    else:
        return gym.make(env_name)


def flatten_observation(obs):
    if isinstance(obs, dict):
        parts = []
        for v in obs.values():
            parts.append(np.asarray(v).ravel())
        return np.concatenate(parts).astype(np.float32)
    else:
        return np.asarray(obs).ravel().astype(np.float32)


def train(
    env_name: str,
    max_steps: int,
    batch_size: int,
    replay_size: int,
    n_step: int,
    gamma: float,
    num_samples: int,
    target_policy_update_period: int,
    target_critic_update_period: int,
    lr: float,
    dual_lr: float,
    device_str: str,
    seed: int,
    eval_freq: int = 0,
    eval_episodes: int = 5,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and device_str == "cuda" else "cpu"
    )

    env = make_env(env_name)
    # separate environment for evaluation (keeps train env state intact)
    eval_env = make_env(env_name) if eval_freq and eval_episodes > 0 else None
    obs0, _ = env.reset()
    obs_flat = flatten_observation(obs0)
    # determine obs and action dims
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Dict):
        obs_dim = sum(int(np.prod(sp.shape)) for sp in obs_space.spaces.values())
    else:
        obs_dim = int(np.prod(obs_space.shape))
    act_space = env.action_space
    action_dim = int(np.prod(act_space.shape))
    action_low = act_space.low
    action_high = act_space.high

    agent = MPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        device=device,
        batch_size=batch_size,
        replay_size=replay_size,
        n_step=n_step,
        gamma=gamma,
        num_samples=num_samples,
        target_policy_update_period=target_policy_update_period,
        target_critic_update_period=target_critic_update_period,
        lr_policy=lr,
        lr_critic=lr,
        lr_dual=dual_lr,
    )

    obs, _ = env.reset()
    obs_flat = flatten_observation(obs)
    episode_return = 0.0
    episode_len = 0
    start_time = time.time()

    for step in range(1, max_steps + 1):
        action = agent.select_action(obs_flat, stochastic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_flat = flatten_observation(next_obs)
        # do not bootstrap from terminal states: use 0.0 discount on terminal steps
        step_discount = 0.0 if done else 1.0
        agent.store_transition(
            obs_flat, action, float(reward), step_discount, next_flat, done
        )
        episode_return += float(reward)
        episode_len += 1
        obs_flat = next_flat

        if done:
            obs, _ = env.reset()
            obs_flat = flatten_observation(obs)
            print(
                f"[Step {step}] Episode finished, return={episode_return:.2f}, len={episode_len}"
            )
            episode_return = 0.0
            episode_len = 0

        stats = agent.learn_step()
        if stats is not None:
            # include episode metrics with the agent stats and log to Weights & Biases
            log_dict = dict(stats)
            log_dict.update(
                {
                    "train/episode_return": episode_return,
                    "train/episode_length": episode_len,
                }
            )
            wandb.log(log_dict, step=step)

        # periodic evaluation
        if eval_env is not None and step % eval_freq == 0:
            eval_returns = []
            eval_lengths = []
            for _ in range(eval_episodes):
                o, _ = eval_env.reset()
                o_flat = flatten_observation(o)
                done = False
                ep_ret = 0.0
                ep_len = 0
                while not done:
                    a = agent.select_action(o_flat, stochastic=False)
                    no, r, terminated, truncated, _ = eval_env.step(a)
                    done = terminated or truncated
                    o_flat = flatten_observation(no)
                    ep_ret += float(r)
                    ep_len += 1
                eval_returns.append(ep_ret)
                eval_lengths.append(ep_len)
            mean_r = float(np.mean(eval_returns))
            std_r = float(np.std(eval_returns))
            mean_len = float(np.mean(eval_lengths))
            eval_log = {
                "eval/mean_reward": mean_r,
                "eval/std_return": std_r,
                "eval/mean_ep_length": mean_len,
            }
            wandb.log(eval_log, step=step)
            print(
                f"[Eval @ step {step}] mean_reward={mean_r:.2f} std={std_r:.2f} mean_len={mean_len:.1f}"
            )

        if stats is not None and step % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"Step {step} | Learn step {stats['train/learn_steps']} | critic_loss={stats['train/critic_loss']:.6f} | Ep Return {episode_return:.2f} | Ep Length {episode_len} | elapsed={elapsed:.1f}s"
            )

    env.close()
    if eval_env is not None:
        eval_env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_names",
        type=str,
        default="cartpole::balance",
        help="Comma-separated list of environment names to train on",
    )
    parser.add_argument("--env_iterations", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--n_step", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--target_policy_update_period", type=int, default=25)
    parser.add_argument("--target_critic_update_period", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dual_lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--static_seed", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default="mpo_project")
    parser.add_argument("--wandb_entity", type=str, default="adrian-research")
    parser.add_argument("--wandb_group_prefix", type=str, default=None)
    parser.add_argument("--base_log_dir", type=str, default="./logs/mpo_experiment")
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=3000,
        help="Evaluate every N environment steps (0 to disable)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=5,
        help="Number of deterministic episodes per evaluation",
    )
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

            seed = (
                torch.randint(0, 10000, (1,)).item()
                if args.static_seed is None
                else args.static_seed
            )

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
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                replay_size=args.replay_size,
                n_step=args.n_step,
                gamma=args.gamma,
                num_samples=args.num_samples,
                target_policy_update_period=args.target_policy_update_period,
                target_critic_update_period=args.target_critic_update_period,
                lr=args.lr,
                dual_lr=args.dual_lr,
                device_str=args.device,
                seed=seed,
                eval_freq=args.eval_freq,
                eval_episodes=args.eval_episodes,
            )

            wandb.finish()
            end_time = time.time()
            print(
                f"Iteration {iteration + 1}/{args.env_iterations} completed in "
                f"{end_time - start_time:.2f} seconds."
            )
