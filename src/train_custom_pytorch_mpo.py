import argparse
import os
import json
import time
import sys
import random
import math
from typing import Tuple

import torch.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium
import numpy as np
import wandb
import imageio
import logging

from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage # type: ignore
from tensordict import TensorDict # type: ignore


LOGGER_NAME = "mpo"
logger = logging.getLogger(LOGGER_NAME)


class MPOConfig:
    def __init__(
        self,
        env_name="HalfCheetah-v5",
        batch_size=256,
        max_actor_steps=1500000,
        num_candidate_actions=32,
        min_replay_size=30_000,
        num_optimization_steps_per_step=2,
        q_lr=0.3e-4,
        pi_lr=0.3e-4,
        tau=0.005,
        dual_lr=1e-3,
        eta=1.0,
        kl_epsilon=0.2,
        epsilon_mean=2.5e-3,
        epsilon_stddev=1e-6,
        temperature_min=1e-3,
        temperature_max=50.0,
        target_q_num_samples=8,
        policy_old_sync_frequency=50,
        log_dir="./logs/mpo_experiment",
        eval_freq=10,
        eval_episodes=5,
        seed=42,
        entropy_coeff=1e-3,
        checkpoint_ep_freq=50,
        e_step_solve_dual=True,
        pi_max_grad_norm=0.5,
        q_max_grad_norm=1.0,
        replay_buffer_size=1000000,
        *args,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.max_actor_steps = max_actor_steps
        self.num_candidate_actions = num_candidate_actions
        self.min_replay_size = min_replay_size
        self.num_optimization_steps_per_step = num_optimization_steps_per_step
        self.q_lr = q_lr
        self.pi_lr = pi_lr
        self.tau = tau
        self.dual_lr = dual_lr
        self.eta = eta
        self.kl_epsilon = kl_epsilon
        self.epsilon_mean = epsilon_mean
        self.epsilon_stddev = epsilon_stddev
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.target_q_num_samples = target_q_num_samples
        self.policy_old_sync_frequency = policy_old_sync_frequency
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.entropy_coeff = entropy_coeff
        self.env_name = env_name
        self.checkpoint_ep_freq = checkpoint_ep_freq
        self.e_step_solve_dual = e_step_solve_dual
        self.pi_max_grad_norm = pi_max_grad_norm
        self.q_max_grad_norm = q_max_grad_norm
        self.replay_buffer_size = replay_buffer_size


class QNetwork(nn.Module):
    """
    MLP Q-network expecting inputs (states, actions).
    Provides:
      - forward(states, actions) -> [B] tensor of Q-values
      - retrace_targets(...) helper to compute Retrace-style targets
    Note: retrace_targets implements a practical, sample-based Retrace-ish
    estimator suitable for use with an n-step replay buffer where only single-step
    behavior log-probs are available. It approximates E_{a'~pi}[Q(s',a')] by
    sampling multiple actions and applies a truncated importance weight
    c = min(1, pi_log_prob - b_log_prob).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        input_dim = obs_dim + act_dim
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # states: [B, obs_dim], actions: [B, act_dim] -> returns [B]
        x = torch.cat([states, actions], dim=-1)
        out = self.net(x).squeeze(-1)
        return out

    @torch.no_grad()
    def retrace_targets(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
        logp_mu: torch.Tensor,
        pi: nn.Module,
        q_target: "QNetwork",
        gamma: float = 0.99,
        num_action_samples: int = 8,
    ):
        """
        Compute Retrace-style targets for a batch.
        Args:
          states, actions, rewards, dones, next_states: tensors shaped [B,...]
          logp_mu: tensor with behavior log-prob for sampled actions [B] (may be zeros)
          pi: current policy (must implement .sample and .log_prob)
          q_target: target Q-network (callable like forward)
          gamma: discount
          num_action_samples: number of actions to sample per next-state for E_pi[Q]
        Returns:
          target_q: tensor [B] for use with MSE loss against q(states,actions)
        Notes:
          - If logp_mu contains zeros (or is not meaningful), importance weights fallback to 1 (clamped).
          - This is a practical sample-based Retrace approximation (one-step corrected).
        """
        device = states.device
        B = states.shape[0]

        # Estimate E_{a'~pi}[ Q_target(next_state, a') ] by sampling multiple actions
        if num_action_samples <= 1:
            # single sample (fast path)
            next_actions, _ = pi.sample(next_states)
            q_next = q_target(next_states, next_actions)
        else:
            # Expand next_states to [B * M, obs_dim]
            next_states_exp = next_states.unsqueeze(1).expand(
                -1, num_action_samples, -1
            )
            next_states_flat = next_states_exp.reshape(B * num_action_samples, -1)
            actions_flat, _ = pi.sample(next_states_flat)
            # Evaluate Q on flattened batch and average per-state
            q_next_flat = q_target(next_states_flat, actions_flat)  # [B*M]
            q_next = q_next_flat.view(B, num_action_samples).mean(dim=1)

        # standard one-step target (expected under pi)
        target = rewards + gamma * (1.0 - dones) * q_next

        # compute truncated importance weight for the sampled actions: c = min(1, exp(log_pi - logp_mu))
        try:
            log_pi_sample = pi.log_prob(states, actions)  # [B]
        except Exception:
            # if policy doesn't support log_prob signature used elsewhere, try flipped args
            log_pi_sample = pi.log_prob(actions, states)

        # Ensure tensors are same dtype/device
        logp_mu_t = (
            logp_mu.to(device=device, dtype=log_pi_sample.dtype)
            if isinstance(logp_mu, torch.Tensor)
            else torch.tensor(logp_mu, device=device, dtype=log_pi_sample.dtype)
        )

        # numeric stability: where behavior log-prob is zero (or extremely small), fall back to weight 1.
        # compute ratio = exp(log_pi - logp_mu) then trunc to 1
        ratio = torch.exp(log_pi_sample - logp_mu_t)
        c = torch.minimum(ratio, torch.ones_like(ratio))

        # Final retrace-style target (one-step truncated correction):
        # target_retrace = q(s,a) + c * (target - q(s,a))
        # Return full target; caller typically computes loss against q(states,actions)
        return target * c + (1.0 - c) * self(states, actions)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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


def setup_logging(log_dir: str, filename: str = "training.log") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def compute_grad_stats(params):
    """Compute L2 grad norm and max-abs grad using torch (returns floats)."""
    total_sq = None
    max_abs = None
    found = False
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        g_sq = (g**2).sum()
        g_max = g.abs().max()
        if not found:
            total_sq = g_sq
            max_abs = g_max
            found = True
        else:
            total_sq = total_sq + g_sq
            max_abs = torch.max(max_abs, g_max)
    if not found:
        return {"grad_norm": 0.0, "grad_max": 0.0}
    grad_norm = torch.sqrt(total_sq).item()
    return {"grad_norm": float(grad_norm), "grad_max": float(max_abs.item())}


def checkpoint_if_needed(
    config: MPOConfig,
    episode: int,
    global_step: int,
    q: QNetwork,
    q_target: QNetwork,
    pi: GaussianPolicy,
    pi_old: GaussianPolicy,
    q_optimizer: torch.optim.Optimizer,
    pi_optimizer: torch.optim.Optimizer,
    checkpoint_max_eval_return: bool = False,
) -> bool:
    if (
        episode + 1
    ) % config.checkpoint_ep_freq != 0 and not checkpoint_max_eval_return:
        wandb.log({"train/checkpoint_ep": 0}, step=global_step)
        return False

    checkpoint_dir = os.path.join(config.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        checkpoint = {
            "episode": episode + 1,
            "global_step": global_step,
            "q_state_dict": q.state_dict(),
            "q_target_state_dict": q_target.state_dict(),
            "pi_state_dict": pi.state_dict(),
            "pi_old_state_dict": pi_old.state_dict(),
            "q_optimizer_state_dict": q_optimizer.state_dict(),
            "pi_optimizer_state_dict": pi_optimizer.state_dict(),
        }
        if checkpoint_max_eval_return:
            checkpoint_name = f"checkpoint_maxeval.pt"
        else:
            checkpoint_name = f"checkpoint_ep{episode+1}.pt"
        ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(checkpoint, ckpt_path)
        torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint_latest.pt"))
        logger.info("Saved checkpoint to %s", ckpt_path)
        wandb.log({"train/checkpoint_ep": episode + 1}, step=global_step)
    except Exception as e:
        logger.error("Failed to save checkpoint: %s", e)
        return False
    return True


def make_offscreen_env(env_name: str):
    """
    Create a gymnasium MuJoCo env with an offscreen-capable backend,
    trying a small set of MUJOCO_GL configurations depending on OS.

    Linux:
        Try MUJOCO_GL in order: 'egl', 'osmesa', 'glfw'.

    macOS:
        First clear MUJOCO_GL (use default Metal/GLFW path),
        then try MUJOCO_GL='glfw' as a fallback.

    Raises RuntimeError with diagnostics if everything fails.
    """
    last_err = None
    platform = sys.platform

    if platform.startswith("linux"):
        backend_candidates = ("egl", "osmesa", "glfw")
    elif platform == "darwin":
        # None = "do not set MUJOCO_GL at all"
        backend_candidates = (None, "glfw")
    else:
        # best-effort generic fallback
        backend_candidates = (None, "egl", "osmesa", "glfw")

    for backend in backend_candidates:
        # Configure MUJOCO_GL for this attempt
        if backend is None:
            os.environ.pop("MUJOCO_GL", None)
            backend_name = "<default>"
        else:
            os.environ["MUJOCO_GL"] = backend
            backend_name = backend

        try:
            env = gymnasium.make(env_name, render_mode="rgb_array")

            # Some envs lazily create the context; force a reset + render
            try:
                obs, _ = env.reset()
                frame = env.render()
                # If the render returns None, just treat it as "not ready yet"
                _ = frame
            except Exception:
                # It's fine; env creation itself worked
                pass

            logger.info(
                "using MUJOCO_GL=%s on platform=%s",
                backend_name,
                platform,
            )
            return env
        except Exception as e:
            last_err = e
            logger.warning(
                "backend %s failed on %s: %s",
                backend_name,
                platform,
                e,
            )
            try:
                env.close()
            except Exception:
                pass

    raise RuntimeError(
        "Failed to initialize an offscreen MuJoCo rendering backend.\n"
        f"Platform: {platform}\n"
        f"Tried backends (in order): {backend_candidates}\n"
        f"Last error: {last_err}\n"
        "On Linux, ensure EGL/OSMesa are installed or try running with a display.\n"
        "On macOS, modern MuJoCo uses Metal; avoid forcing MUJOCO_GL to EGL/OSMesa."
    )


def generate_video(
    env_name: str,
    policy: torch.nn.Module,
    output_path: str,
    num_episodes: int = 1,
    max_steps: int = 1000,
    fps: int = 30,
    deterministic: bool = False,
    env: gymnasium.Env | None = None,
):
    if env is None:
        env = make_offscreen_env(env_name)
    obs0, _ = env.reset()
    writer = imageio.get_writer(output_path, fps=fps)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    try:
        logger.info("Recording %d episodes to %s...", num_episodes, output_path)
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            # render initial frame (some envs require rendering after reset)
            try:
                frame = env.render()
            except Exception as e:
                frame = None
                logger.warning("initial render failed: %s", e)
            if frame is not None:
                # ensure uint8
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                writer.append_data(frame)

            while not done and steps < max_steps:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    if deterministic:
                        mu, _ = policy(obs_t)
                        a_tanh = torch.tanh(mu)
                        action_t = (
                            (a_tanh * policy.action_scale + policy.action_bias)
                            .cpu()
                            .numpy()[0]
                        )
                    else:
                        action_t, _ = policy.sample(obs_t)
                        action_t = action_t.cpu().numpy()[0]

                next_obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated
                obs = next_obs

                try:
                    frame = env.render()
                except Exception as e:
                    frame = None
                    logger.warning("render failed at step %d: %s", steps, e)

                if frame is not None:
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                    writer.append_data(frame)

                steps += 1

            logger.info("Episode %d recorded, steps=%d", ep + 1, steps)
    finally:
        writer.close()
        env.close()
    logger.info("Saved video to: %s", output_path)


def load_policy_from_checkpoint(ckpt_path: str, policy: torch.nn.Module):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Prefer 'pi_state_dict' then 'pi_old_state_dict'
    if "pi_state_dict" in ckpt:
        state = ckpt["pi_state_dict"]
    elif "pi_old_state_dict" in ckpt:
        state = ckpt["pi_old_state_dict"]
    else:
        # attempt to use top-level state_dict if present
        state = ckpt.get("state_dict", ckpt)
    policy.load_state_dict(state)
    return ckpt


def evaluate_policy(policy: GaussianPolicy, env, device, n_eval_episodes: int = 5):
    """
    Run the policy for n_eval_episodes (stochastic sampling) and return list of episode returns and episode lengths.
    """
    returns = []
    lengths = []
    prior_mode = policy.training
    policy.eval()
    with torch.inference_mode():
        for _ in range(n_eval_episodes):
            obs = torch.tensor(
                env.reset()[0], dtype=torch.float32, device=device
            ).unsqueeze(0)
            done = False
            ep_ret = 0.0
            ep_len = 0
            while not done:
                action_tensor, _ = policy.sample(obs)
                action = action_tensor.cpu().numpy()[0]
                next_obs, reward, terminated, truncated, _ = env.step(action)
                ep_ret += float(reward)
                ep_len += 1
                done = terminated or truncated
                obs = torch.tensor(
                    next_obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
            returns.append(ep_ret)
            lengths.append(ep_len)
    policy.train(prior_mode)
    return returns, lengths


def compute_weights_and_temperature_loss_torch(
    q_values: torch.Tensor,
    epsilon: float,
    temperature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch equivalent of Acme's compute_weights_and_temperature_loss.

    Args:
      q_values: [K, B] Q-values for K sampled actions per state.
      epsilon: scalar KL constraint target.
      temperature: positive scalar dual variable (already in primal space).

    Returns:
      normalized_weights: [K, B] softmax-normalized weights (no grad).
      temperature_loss: scalar temperature (dual) loss for backprop.
    """
    # Ensure we don't propagate gradients through Q into the temperature update.
    tempered_q = (q_values.detach()) / temperature

    # Softmax over action-sample dimension K.
    normalized_weights = torch.softmax(tempered_q, dim=0).detach()

    # Dual loss: epsilon + E[logsumexp(Q/τ) - log K], multiplied by τ.
    # dim=0 -> over K actions; mean over batch B.
    q_logsumexp = torch.logsumexp(tempered_q, dim=0)  # [B]
    log_num_actions = math.log(q_values.shape[0])
    loss_temperature_inner = float(epsilon) + q_logsumexp.mean() - log_num_actions
    temperature_loss = temperature * loss_temperature_inner
    return normalized_weights, temperature_loss


def compute_nonparametric_kl_from_normalized_weights_torch(
    normalized_weights: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch equivalent of Acme's compute_nonparametric_kl_from_normalized_weights.

    Args:
      normalized_weights: [K, B] normalized discrete distribution over K actions.

    Returns:
      kl: [B] estimated KL(q_nonparametric || uniform over K actions).
    """
    K = normalized_weights.shape[0]
    num_action_samples = float(K)
    integrand = torch.log(num_action_samples * normalized_weights + 1e-8)
    return torch.sum(normalized_weights * integrand, dim=0)


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


def policy_evaluation_e_step(
    states: torch.Tensor,
    pi_old: nn.Module,
    q: nn.Module,
    K: int,
    temperature: torch.Tensor,
    epsilon: float,
):
    """
    E-step: sample K actions per state and construct a non-parametric policy
    via Acme-style tempered Q / temperature dual.

    Args:
      states: [B, obs_dim]
      pi_old: behavior / target policy (GaussianPolicy)
      q: Q-network
      K: number of action samples per state
      temperature: positive scalar tensor (after softplus of log-temperature)
      epsilon: KL constraint target for the E-step weights.
    Returns:
      actions: [B, K, act_dim] sampled from pi_old
      q_dist: [B, K] normalized weights (non-parametric policy)
      kl_np: float, estimated discrete KL(q_nonparametric || π_old over samples)
      temperature_loss: scalar dual loss for updating log-temperature
      temperature_value: float temperature (for logging)
    """
    B, obs_dim = states.shape

    with torch.no_grad():
        states_expanded = states.unsqueeze(1).expand(-1, K, -1)
        states_flat = states_expanded.reshape(B * K, obs_dim)

        # Sample actions under pi_old, then re-evaluate log-probs for KL stats.
        (
            actions_flat,
            log_pi_flat,
            _,
            pre_tanh_flat,
        ) = pi_old._sample_pre_tanh(states_flat)
        log_pi_flat = log_pi_flat  # [B*K]

        q_flat = q(states_flat, actions_flat)  # [B*K]

        actions = actions_flat.view(B, K, -1)
        q_vals = q_flat.view(K, B)  # [K, B] for helper, transpose below as needed
        log_pi_old = log_pi_flat.view(B, K)

    # Use Acme-style helper (no grad into Q) to get non-parametric weights
    # and the dual loss for the temperature. q_values is [K,B]; our q_vals is
    # [K,B] already, so this is consistent.
    normalized_weights, temperature_loss = compute_weights_and_temperature_loss_torch(
        q_values=q_vals,
        epsilon=float(epsilon),
        temperature=temperature,
    )

    # Transpose back to [B,K] to align with rest of code.
    q_dist = normalized_weights.t()  # [B, K]

    # KL diagnostic between non-parametric q and discrete π_old over samples.
    with torch.no_grad():
        # Discrete π_old over the candidate set (per-state normalization).
        log_pi_old_norm = log_pi_old - torch.logsumexp(log_pi_old, dim=-1, keepdim=True)
        pi_old_disc = torch.exp(log_pi_old_norm)  # [B, K]

        # Non-parametric KL using helper; returns [B]
        kl_np_vec = compute_nonparametric_kl_from_normalized_weights_torch(
            normalized_weights
        )  # [B], still in [K,B] convention
        kl_np = float(kl_np_vec.mean().item())

    return (
        actions.detach(),
        q_dist,
        kl_np,
        temperature_loss,
        float(temperature.item()),
    )


def policy_evaluation_m_step(
    policy_net: GaussianPolicy,
    target_policy_net: GaussianPolicy,
    states: torch.Tensor,
    actions: torch.Tensor,
    weights: torch.Tensor,
    entropy_coeff: float = 0.0,
    alpha_mean: torch.Tensor | None = None,
    alpha_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Continuous-action M-step:
    Fit pi_theta(a|s) to q(a|s) via weighted log-likelihood.
    We compute weighted log-likelihood per state (sum over K) and then average over batch.
    Inputs:
      states:  [B, obs_dim]
      actions: [B, K, act_dim]
      weights: [B, K]
    """
    B, K, act_dim = actions.shape
    obs_dim = states.shape[-1]

    # Expand and flatten: [B, obs_dim] -> [B, K, obs_dim] -> [B*K, obs_dim]
    states_expanded = states.unsqueeze(1).expand(-1, K, -1).reshape(B * K, obs_dim)
    actions_flat = actions.reshape(B * K, act_dim)
    log_pi = policy_net.log_prob(states_expanded, actions_flat)  # [B*K]

    log_pi_per = log_pi.view(B, K)
    # Weighted sum per-state, then average over states
    weighted_ll_per_state = (weights.detach() * log_pi_per).sum(dim=1)  # [B]
    loss_pi = -weighted_ll_per_state.mean()

    ent = None
    if entropy_coeff:
        mu_state, log_std_state = policy_net(states)
        ent_const = 0.5 * (
            1.0
            + torch.log(
                torch.tensor(
                    2.0 * math.pi,
                    device=log_std_state.device,
                    dtype=log_std_state.dtype,
                )
            )
        )
        ent = (ent_const + log_std_state).sum(dim=-1).mean()
        loss_pi = loss_pi - entropy_coeff * ent
    else:
        mu_state, log_std_state = policy_net(states)

    with torch.no_grad():
        target_mu, target_log_std = target_policy_net(states)
    target_std = torch.exp(target_log_std)
    policy_std = torch.exp(log_std_state)

    # KL decomposition using torch distributions.
    kl_mean = dist.kl_divergence(
        dist.Normal(target_mu, target_std), dist.Normal(mu_state, target_std)
    ).sum(
        dim=-1
    )  # [B]
    kl_std = dist.kl_divergence(
        dist.Normal(target_mu, target_std), dist.Normal(target_mu, policy_std)
    ).sum(
        dim=-1
    )  # [B]
    loss_kl_mean = torch.zeros((), device=states.device, dtype=states.dtype)
    loss_kl_std = torch.zeros((), device=states.device, dtype=states.dtype)
    if alpha_mean is not None:
        loss_kl_mean = alpha_mean.detach() * kl_mean.mean()
        loss_pi = loss_pi + loss_kl_mean
    if alpha_std is not None:
        loss_kl_std = alpha_std.detach() * kl_std.mean()
        loss_pi = loss_pi + loss_kl_std

    stats = {
        "kl_mean": kl_mean.mean().detach(),
        "kl_std": kl_std.mean().detach(),
    }
    if ent is not None:
        stats["entropy"] = ent.detach()

    return loss_pi, stats


def warmup_replay_buffer(
    env: gymnasium.Env, device: torch.device, config: MPOConfig, pi_old: nn.Module
) -> TensorDictReplayBuffer:
    storage = LazyTensorStorage(config.replay_buffer_size, device=device)
    replay_buffer = TensorDictReplayBuffer(storage=storage)
    obs, _ = env.reset(seed=config.seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    while len(replay_buffer) < config.min_replay_size:
        with torch.no_grad():
            action_tensor, _ = pi_old.sample(obs)
            action = action_tensor.cpu().numpy()[0]
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        transition = TensorDict(
            {
                "obs": obs.squeeze(0),
                "action": torch.tensor(action, device=device, dtype=torch.float32),
                "reward": torch.tensor(reward, device=device, dtype=torch.float32),
                "done": torch.tensor(done, device=device, dtype=torch.bool),
                "next_obs": torch.tensor(next_obs, dtype=torch.float32, device=device),
                "logp_mu": torch.tensor(0.0, device=device, dtype=torch.float32),
            },
            batch_size=[],
        )
        replay_buffer.add(transition)

        if done:
            next_obs, _ = env.reset()
        obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
    return replay_buffer


def train_mpo(config: MPOConfig, device: torch.device) -> GaussianPolicy:
    env = gymnasium.make(config.env_name)

    eval_env = gymnasium.make(config.env_name)
    eval_env.reset(seed=config.seed + 1007)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    q = QNetwork(obs_dim, act_dim).to(device)
    q_target = QNetwork(obs_dim, act_dim).to(device)
    q_target.load_state_dict(q.state_dict())

    pi = GaussianPolicy(obs_dim, act_dim, action_low, action_high).to(device)
    pi_old = GaussianPolicy(obs_dim, act_dim, action_low, action_high).to(device)
    pi_old.load_state_dict(pi.state_dict())

    if wandb.run:
        wandb.run.summary["model/q_network"] = str(q)
        wandb.run.summary["model/policy_network"] = str(pi)

    q_optimizer = torch.optim.Adam(q.parameters(), lr=config.q_lr)
    pi_optimizer = torch.optim.Adam(pi.parameters(), lr=config.pi_lr)

    # Centralize MPO duals in MPOLoss and attach an optimizer.
    mpo_loss_module = MPOLoss(eta=config.eta, device=device)
    dual_optimizer = torch.optim.Adam(mpo_loss_module.parameters(), lr=config.dual_lr)

    replay_buffer = warmup_replay_buffer(env, device, config, pi_old)

    logger = setup_logging(config.log_dir)

    global_step = 0
    max_eval_return = -float("inf")
    episode = 0
    while True:
        episode += 1
        if global_step >= config.max_actor_steps:
            break
        logger.info("Starting episode %d ...", episode)
        wandb.log({"train/episode": episode}, step=global_step)

        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        ep_reward = 0.0
        ep_length = 0
        while not done:
            ep_length += 1
            with torch.no_grad():
                action_tensor, _ = pi_old.sample(obs)
                action = action_tensor.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)

            transition = TensorDict(
                {
                    "obs": obs.squeeze(0),
                    "action": torch.tensor(action, device=device, dtype=torch.float32),
                    "reward": torch.tensor(reward, device=device, dtype=torch.float32),
                    "done": torch.tensor(done, device=device, dtype=torch.bool),
                    "next_obs": torch.tensor(
                        next_obs, dtype=torch.float32, device=device
                    ),
                    "logp_mu": torch.tensor(0.0, device=device, dtype=torch.float32),
                },
                batch_size=[],
            )
            replay_buffer.add(transition)
            global_step += 1
            wandb.log({"train/global_step": global_step}, step=global_step)

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(
                0
            )
            # Run a block of optimization steps (each step samples a mini-batch)
            for opt_iter in range(config.num_optimization_steps_per_step):
                # Sample a mini-batch B of N (s, a, r) pairs from replay
                batch = replay_buffer.sample(config.batch_size)
                states = batch["obs"]
                acts = batch["action"]
                rewards = batch["reward"]
                dones = batch["done"].float()
                next_states = batch["next_obs"]
                logp_mu = batch["logp_mu"]

                q_sa = q(states, acts)

                with torch.no_grad():
                    if config.target_q_num_samples <= 1:
                        next_actions, _ = pi_old.sample(next_states)
                        q_next = q_target(next_states, next_actions)
                    else:
                        B = next_states.shape[0]
                        next_states_expanded = next_states.unsqueeze(1).expand(
                            -1, config.target_q_num_samples, -1
                        )
                        next_states_flat = next_states_expanded.reshape(
                            B * config.target_q_num_samples, -1
                        )
                        sampled_actions_flat, _ = pi_old.sample(next_states_flat)
                        q_next_flat = q_target(next_states_flat, sampled_actions_flat)
                        q_next = q_next_flat.view(config.target_q_num_samples, B).mean(
                            dim=0
                        )
                    target_q = rewards + 0.99 * (1.0 - dones) * q_next

                loss_q = F.mse_loss(q_sa, target_q)

                q_optimizer.zero_grad()
                loss_q.backward()

                q_grad_stats = compute_grad_stats(q.parameters())
                wandb.log(
                    {
                        "train/grad_norm_q": q_grad_stats["grad_norm"],
                        "train/grad_max_q": q_grad_stats["grad_max"],
                    },
                    step=global_step,
                )

                if config.q_max_grad_norm is not None and config.q_max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        q.parameters(), config.q_max_grad_norm
                    )

                q_grad_stats_clipped = compute_grad_stats(q.parameters())
                wandb.log(
                    {
                        "train/grad_norm_q_clipped": q_grad_stats_clipped["grad_norm"],
                        "train/grad_max_q_clipped": q_grad_stats_clipped["grad_max"],
                    },
                    step=global_step,
                )

                q_optimizer.step()

                for tp, p in zip(q_target.parameters(), q.parameters()):
                    tp.data.mul_(1.0 - config.tau).add_(config.tau * p.data)

                # E-step: get primal temperature & dual alphas from module.
                temperature = mpo_loss_module.temperature()
                temperature = torch.clamp(
                    temperature, config.temperature_min, config.temperature_max
                )
                alpha_mean, alpha_std = mpo_loss_module.alphas()
                (actions_e, q_dist, kl_np, temperature_loss, _) = (
                    policy_evaluation_e_step(
                        states,
                        pi_old,
                        q,
                        config.num_candidate_actions,
                        temperature=temperature,
                        epsilon=config.kl_epsilon,
                    )
                )
                pi_loss, policy_stats = policy_evaluation_m_step(
                    pi,
                    pi_old,
                    states,
                    actions_e,
                    q_dist,
                    entropy_coeff=config.entropy_coeff,
                    alpha_mean=alpha_mean.detach(),
                    alpha_std=alpha_std.detach(),
                )
                kl_mean_det = policy_stats["kl_mean"]
                kl_std_det = policy_stats["kl_std"]

                # Form dual loss and update dual parameters from mpo_loss_module.
                dual_optimizer.zero_grad()
                dual_loss = (
                    temperature_loss
                    - alpha_mean * (kl_mean_det - config.epsilon_mean)
                    - alpha_std * (kl_std_det - config.epsilon_stddev)
                )
                dual_loss_value = float(dual_loss.detach().cpu().item())
                dual_loss.backward(retain_graph=True)
                dual_optimizer.step()

                pi_optimizer.zero_grad()
                pi_loss.backward()

                pi_grad_stats = compute_grad_stats(pi.parameters())
                wandb.log(
                    {
                        "train/grad_norm_pi": pi_grad_stats["grad_norm"],
                        "train/grad_max_pi": pi_grad_stats["grad_max"],
                    },
                    step=global_step,
                )

                if (
                    config.pi_max_grad_norm is not None
                    and config.pi_max_grad_norm > 0.0
                ):
                    torch.nn.utils.clip_grad_norm_(
                        pi.parameters(), config.pi_max_grad_norm
                    )

                pi_grad_stats_clipped = compute_grad_stats(pi.parameters())
                wandb.log(
                    {
                        "train/grad_norm_pi_clipped": pi_grad_stats_clipped[
                            "grad_norm"
                        ],
                        "train/grad_max_pi_clipped": pi_grad_stats_clipped["grad_max"],
                    },
                    step=global_step,
                )

                pi_optimizer.step()
                current_temperature = mpo_loss_module.temperature().detach()
                current_temperature = torch.clamp(
                    current_temperature, config.temperature_min, config.temperature_max
                )
                current_alpha_mean = mpo_loss_module.alphas()[0].detach()
                current_alpha_std = mpo_loss_module.alphas()[1].detach()

            log_payload = {
                "train/learning_rate_q": q_optimizer.param_groups[0]["lr"],
                "train/learning_rate_pi": pi_optimizer.param_groups[0]["lr"],
                "train/q_sa": q_sa.mean().item(),
                "train/loss_q": loss_q.item(),
                "train/kl_np": kl_np,
                "train/temperature": float(current_temperature.cpu().item()),
                "train/temperature_loss": float(temperature_loss.detach().cpu().item()),
                "train/log_temperature": float(
                    mpo_loss_module.log_temperature.detach().cpu().item()
                ),
                "train/pi_loss": pi_loss.item(),
                "train/dual_loss": dual_loss_value,
                "train/alpha_mean": float(current_alpha_mean.cpu().item()),
                "train/alpha_std": float(current_alpha_std.cpu().item()),
                "train/log_alpha_mean": float(
                    mpo_loss_module.log_alpha_mean.detach().cpu().item()
                ),
                "train/log_alpha_stddev": float(
                    mpo_loss_module.log_alpha_stddev.detach().cpu().item()
                ),
                "train/kl_mean": float(kl_mean_det.cpu().item()),
                "train/kl_std": float(kl_std_det.cpu().item()),
            }
            if "entropy" in policy_stats:
                log_payload["train/policy_entropy"] = float(
                    policy_stats["entropy"].cpu().item()
                )
            wandb.log(log_payload, step=global_step)
        if config.policy_old_sync_frequency > 0:
            if global_step % config.policy_old_sync_frequency == 0:
                pi_old.load_state_dict(pi.state_dict())
                sync_flag = 1.0
            else:
                sync_flag = 0.0
            wandb.log({"train/policy_old_synced": sync_flag}, step=global_step)
        wandb.log(
            {"train/ep_length": ep_length, "train/ep_reward": ep_reward},
            step=global_step,
        )
        logger.info(
            "episode=%d global_step=%d ep_length=%.3f",
            episode,
            global_step,
            ep_length,
        )

        checkpoint_max_eval_return = False
        if episode % config.eval_freq == 0:
            eval_returns, eval_lengths = evaluate_policy(
                pi, eval_env, device, n_eval_episodes=config.eval_episodes
            )
            eval_mean = float(np.mean(eval_returns))
            if eval_mean > max_eval_return:
                max_eval_return = eval_mean
                checkpoint_max_eval_return = True
            eval_length_mean = float(np.mean(eval_lengths))
            wandb.log(
                {
                    "eval/mean_reward": eval_mean,
                    "eval/mean_ep_length": eval_length_mean,
                },
                step=global_step,
            )
            logger.info(
                "Eval: episode=%d global_step=%d eval_mean=%.3f eval_length_mean=%.3f",
                episode,
                global_step,
                eval_mean,
                eval_length_mean,
            )

        checkpoint_if_needed(
            config=config,
            checkpoint_max_eval_return=checkpoint_max_eval_return,
            episode=episode,
            global_step=global_step,
            q=q,
            q_target=q_target,
            pi=pi,
            pi_old=pi_old,
            q_optimizer=q_optimizer,
            pi_optimizer=pi_optimizer,
        )

    return pi


def main():
    parser = argparse.ArgumentParser(description="Train MPO")
    parser.add_argument(
        "--env_names",
        type=str,
        default="HalfCheetah-v5",
        help="Comma-separated list of environment names to train on",
    )
    parser.add_argument("--env_iterations", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_actor_steps", type=int, default=1500000)
    parser.add_argument("--num_candidate_actions", type=int, default=32)
    parser.add_argument("--min_replay_size", type=int, default=1000)
    parser.add_argument("--num_optimization_steps_per_step", type=int, default=2)
    parser.add_argument("--q_lr", type=float, default=0.0005)
    parser.add_argument("--pi_lr", type=float, default=0.0005)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--dual_lr", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--kl_epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_mean", type=float, default=2.5e-3)
    parser.add_argument("--epsilon_stddev", type=float, default=1e-6)
    parser.add_argument("--temperature_min", type=float, default=1e-3)
    parser.add_argument("--temperature_max", type=float, default=50.0)
    parser.add_argument("--target_q_num_samples", type=int, default=8)
    parser.add_argument("--policy_old_sync_frequency", type=int, default=100)
    parser.add_argument("--base_log_dir", type=str, default="./logs/mpo_experiment")
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--entropy_coeff", type=float, default=1e-3)
    parser.add_argument("--e_step_solve_dual", type=bool, default=True)
    parser.add_argument("--checkpoint_ep_freq", type=int, default=100)
    parser.add_argument("--pi_max_grad_norm", type=float, default=0.5)
    parser.add_argument("--q_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--replay_buffer_size", type=int, default=1000000)
    parser.add_argument("--wandb_project", type=str, default="mpo_project")
    parser.add_argument("--wandb_entity", type=str, default="adrian-research")
    parser.add_argument("--wandb_group_prefix", type=str, default=None)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    env_names = [name.strip() for name in args.env_names.split(",")]
    for env_name in env_names:
        for iteration in range(args.env_iterations):
            print(
                f"Training on environment: {env_name}. Starting iteration {iteration + 1}/{args.env_iterations}"
            )
            start_time = time.time()

            seed = int(time.time()) % 10000

            experiment_identifier = (
                env_name
                + "__iter"
                + str(iteration + 1)
                + f"_seed{seed}"
                + "_"
                + time.strftime("%Y%m%d-%H%M%S")
            )

            config = MPOConfig(
                env_name=env_name,
                seed=seed,
                log_dir=os.path.join(
                    args.base_log_dir, args.wandb_project + "_" + experiment_identifier
                ),
                **vars(args),
            )
            os.makedirs(config.log_dir, exist_ok=True)

            print("Experiment Configuration:")
            print(json.dumps(vars(config), indent=4))

            with open(os.path.join(config.log_dir, "config.json"), "w") as f:
                json.dump(vars(config), f, indent=4)
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
                dir=config.log_dir,
            )

            train_mpo(config, device)

            try:
                env = make_offscreen_env(env_name)
                policy = GaussianPolicy(
                    env.observation_space.shape[0],
                    env.action_space.shape[0],
                    env.action_space.low,
                    env.action_space.high,
                )
                policy.eval()
                ckpt_path = os.path.join(
                    config.log_dir, "checkpoints", "checkpoint_maxeval.pt"
                )
                print(f"Loading policy from checkpoint: {ckpt_path}")
                with torch.inference_mode():
                    load_policy_from_checkpoint(ckpt_path, policy)
                    generate_video(
                        env_name=env_name,
                        env=env,
                        policy=policy,
                        output_path=os.path.join(config.log_dir, "video.mp4"),
                        num_episodes=2,
                        max_steps=1000,
                        fps=30,
                        deterministic=False,
                    )
            except Exception as e:
                print(f"[ERROR] Warning: video generation failed: {e}")

            wandb.finish()
            end_time = time.time()
            print(
                f"Iteration {iteration + 1}/{args.env_iterations} completed in "
                f"{end_time - start_time:.2f} seconds."
            )


if __name__ == "__main__":
    main()
