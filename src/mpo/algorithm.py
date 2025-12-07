import random
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium
import numpy as np
import wandb

from mpo.gaussian_policy import GaussianPolicy
from mpo.replay_buffer import NStepReplayBuffer
from mpo.q_network import QNetwork
from mpo.utils import (
    evaluate_policy,
    checkpoint_if_needed,
    compute_grad_stats,
    setup_logging,
    compute_weights_and_temperature_loss_torch,
    compute_nonparametric_kl_from_normalized_weights_torch,
)
from mpo.mpo_config import MPOConfig


def policy_evaluation_e_step(
    states: torch.Tensor,
    pi_old: nn.Module,
    q: nn.Module,
    K: int,
    temperature: torch.Tensor,
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
        actions_flat, _ = pi_old.sample(states_flat)
        log_pi_flat = pi_old.log_prob(states_flat, actions_flat)  # [B*K]

        q_flat = q(states_flat, actions_flat)  # [B*K]

        actions = actions_flat.view(B, K, -1)
        q_vals = q_flat.view(K, B)  # [K, B] for helper, transpose below as needed
        log_pi_old = log_pi_flat.view(B, K)

    # Use Acme-style helper (no grad into Q) to get non-parametric weights
    # and the dual loss for the temperature. q_values is [K,B]; our q_vals is
    # [K,B] already, so this is consistent.
    normalized_weights, temperature_loss = compute_weights_and_temperature_loss_torch(
        q_values=q_vals,
        epsilon=float(1.0),  # actual epsilon handled via config in training loop
        temperature=temperature,
    )

    # Transpose back to [B,K] to align with rest of code.
    q_dist = normalized_weights.t()  # [B, K]

    # KL diagnostic between non-parametric q and discrete π_old over samples.
    with torch.no_grad():
        # Discrete π_old over the candidate set (per-state normalization).
        log_pi_old_norm = log_pi_old - torch.logsumexp(
            log_pi_old, dim=-1, keepdim=True
        )
        pi_old_disc = torch.exp(log_pi_old_norm)  # [B, K]

        # Non-parametric KL using helper; returns [B]
        kl_np_vec = compute_nonparametric_kl_from_normalized_weights_torch(
            normalized_weights
        )  # [B], still in [K,B] convention
        kl_np = float(kl_np_vec.mean().item())

    return actions.detach(), q_dist, kl_np, temperature_loss, float(temperature.item())


def policy_evaluation_m_step(
    policy_net: GaussianPolicy,
    states: torch.Tensor,
    actions: torch.Tensor,
    weights: torch.Tensor,
    entropy_coeff: float = 0.0,
) -> torch.Tensor:
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

    # Entropy regularization (approximate using pre-tanh Gaussian entropy)
    if entropy_coeff is not None and entropy_coeff != 0.0:
        # policy_net.forward returns (mu, log_std) for pre-tanh Gaussian
        mu, log_std = policy_net(states_expanded)
        # Entropy per dim of Normal: 0.5 * (1 + log(2*pi)) + log_std
        ent_const = 0.5 * (
            1.0
            + torch.log(
                torch.tensor(2.0 * math.pi, device=log_std.device, dtype=log_std.dtype)
            )
        )
        ent_per_dim = ent_const + log_std
        ent = ent_per_dim.sum(dim=-1).mean()  # scalar
        entropy_bonus = -entropy_coeff * ent
        loss_pi = loss_pi + entropy_bonus

    return loss_pi


def warmup_replay_buffer(
    env: gymnasium.Env, device: torch.device, config: MPOConfig, pi_old: nn.Module
) -> NStepReplayBuffer:
    replay_buffer = NStepReplayBuffer(
        capacity=config.replay_buffer_size,
        obs_shape=env.observation_space.shape,
        act_shape=env.action_space.shape,
        n_step=5,
        gamma=0.99,
        device=device,
    )
    obs, _ = env.reset(seed=config.seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    while len(replay_buffer) < config.min_replay_size:
        with torch.no_grad():
            action_tensor, _ = pi_old.sample(obs)
            action = action_tensor.cpu().numpy()[0]
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(obs.cpu().numpy()[0], action, reward, done, 0.0, next_obs)

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

    # Dual variable: log-temperature (scalar) with softplus transform.
    # Initialize near log(config.eta) to keep behavior similar to original.
    init_log_temperature = math.log(max(config.eta, 1e-3))
    log_temperature = torch.nn.Parameter(
        torch.tensor([init_log_temperature], dtype=torch.float32, device=device)
    )
    dual_optimizer = torch.optim.Adam([log_temperature], lr=config.dual_lr)

    replay_buffer = warmup_replay_buffer(env, device, config, pi_old)

    logger = setup_logging(config.log_dir)

    global_step = 0
    max_eval_return = -float("inf")
    for episode in range(config.num_training_episodes):
        logger.info("Starting episode %d ...", episode + 1)
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

            replay_buffer.push(
                obs.cpu().numpy()[0], action, reward, done, 0.0, next_obs
            )
            global_step += 1
            wandb.log({"train/global_step": global_step}, step=global_step)

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(
                0
            )
            # Run a block of optimization steps (each step samples a mini-batch)
            for opt_iter in range(config.num_optimization_steps_per_step):
                # Sample a mini-batch B of N (s, a, r) pairs from replay
                states, acts, rewards, dones, next_states, logp_mu = (
                    replay_buffer.sample(config.batch_size)
                )

                q_sa = q(states, acts)

                with torch.no_grad():
                    next_actions, _ = pi.sample(next_states)
                    q_next = q_target(next_states, next_actions)
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

                # E-step with learnable temperature (dual variable).
                temperature = torch.nn.functional.softplus(log_temperature) + 1e-8
                # Avoid extremely small or huge temperature values
                temperature = F.softplus(log_temperature) + 1e-8
                temperature = torch.clamp(temperature, 1e-3, 50.0)
                actions_e, q_dist, kl_np, temperature_loss, temperature_value = (
                    policy_evaluation_e_step(
                        states,
                        pi_old,
                        q,
                        config.num_candidate_actions,
                        temperature=temperature,
                    )
                )

                # Update dual variable (temperature) via its own optimizer.
                dual_optimizer.zero_grad()
                temperature_loss.backward(retain_graph=True)
                dual_optimizer.step()

                pi_loss = policy_evaluation_m_step(
                    pi,
                    states,
                    actions_e,
                    q_dist,
                    entropy_coeff=config.entropy_coeff,
                )

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

            wandb.log(
                {
                    "train/learning_rate_q": q_optimizer.param_groups[0]["lr"],
                    "train/learning_rate_pi": pi_optimizer.param_groups[0]["lr"],
                    "train/q_sa": q_sa.mean().item(),
                    "train/loss_q": loss_q.item(),
                    "train/kl_np": kl_np,
                    "train/temperature": temperature_value,
                    "train/temperature_loss": float(temperature_loss.item()),
                    "train/log_temperature": float(log_temperature.detach().cpu().item()),
                    "train/pi_loss": pi_loss.item(),
                },
                step=global_step,
            )
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
            episode + 1,
            global_step,
            ep_length,
        )

        checkpoint_max_eval_return = False
        if (episode + 1) % config.eval_freq == 0:
            eval_returns = evaluate_policy(
                pi, eval_env, device, n_eval_episodes=config.eval_episodes
            )
            eval_mean = float(np.mean(eval_returns))
            if eval_mean > max_eval_return:
                max_eval_return = eval_mean
                checkpoint_max_eval_return = True
            eval_length_mean = float(np.mean([len(r) for r in eval_returns]))
            wandb.log(
                {
                    "eval/mean_reward": eval_mean,
                    "eval/mean_ep_length": eval_length_mean,
                },
                step=global_step,
            )
            logger.info(
                "Eval: episode=%d global_step=%d eval_mean=%.3f eval_length_mean=%.3f",
                episode + 1,
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
