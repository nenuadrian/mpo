import argparse
import random
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb

from gaussian_policy import GaussianPolicy
from replay_buffer import NStepReplayBuffer
from q_network import QNetwork


def policy_evaluation_e_step(
    states: torch.Tensor,
    pi_old: nn.Module,
    q: nn.Module,
    K: int,
    eta: float,
    solve_dual: bool = False,
    kl_target: float = None,
    eta_bounds=(1e-8, 1e6),
    max_iters: int = 40,
    tol: float = 1e-4,
):
    """
    E-step: sample K actions per state, compute q_dist ∝ π_old(a_k) * exp(Q/eta).
    If solve_dual is True, binary-search eta so that avg KL(q || π_old) ≈ kl_target.
    Returns: actions [B,K,act_dim], q_dist [B,K], kl (float), eta (float)
    """
    B, obs_dim = states.shape

    with torch.no_grad():
        states_expanded = states.unsqueeze(1).expand(-1, K, -1)
        states_flat = states_expanded.reshape(B * K, obs_dim)

        # Sample actions but re-evaluate log-prob via log_prob to ensure tanh correction is applied.
        actions_flat, _ = pi_old.sample(states_flat)
        log_pi_flat = pi_old.log_prob(states_flat, actions_flat)  # [B*K]

        q_flat = q(states_flat, actions_flat)

        actions = actions_flat.view(B, K, -1)
        q_vals = q_flat.view(B, K)
        log_pi_old = log_pi_flat.view(B, K)

        # Center Q per state for numerical stability
        q_centered = q_vals - q_vals.max(dim=-1, keepdim=True).values

        # Normalize π_old over SAME candidate set
        log_pi_old_norm = log_pi_old - torch.logsumexp(log_pi_old, dim=-1, keepdim=True)

        def compute_q_dist_and_kl(local_eta):
            # compute discrete q over candidates for given eta
            log_q_tilde = log_pi_old + q_centered / local_eta  # [B, K]
            log_q = log_q_tilde - torch.logsumexp(log_q_tilde, dim=-1, keepdim=True)
            q_dist_local = torch.exp(log_q)  # [B, K]
            # discrete KL(q || π_old) per state, then mean over states
            kl_per_state = torch.sum(q_dist_local * (log_q - log_pi_old_norm), dim=-1)
            kl_mean = kl_per_state.mean().item()
            return q_dist_local, kl_mean

        # If requested, solve for eta using binary search so avg KL ≈ kl_target
        solved_eta = float(eta)
        if solve_dual:
            assert (
                kl_target is not None
            ), "kl_target must be provided when solve_dual=True"
            lo, hi = float(eta_bounds[0]), float(eta_bounds[1])

            # Evaluate KL at lo and hi
            _, kl_lo = compute_q_dist_and_kl(lo)
            _, kl_hi = compute_q_dist_and_kl(hi)

            # If kl_lo already <= target, choose lo; if kl_hi > target try expanding hi
            if kl_lo <= kl_target:
                solved_eta = lo
            else:
                # expand hi until KL(hi) <= target or hit cap
                expand_iters = 0
                while kl_hi > kl_target and expand_iters < 10:
                    hi *= 10.0
                    _, kl_hi = compute_q_dist_and_kl(hi)
                    expand_iters += 1
                # Binary search in [lo, hi]
                for _ in range(max_iters):
                    mid = 0.5 * (lo + hi)
                    _, kl_mid = compute_q_dist_and_kl(mid)
                    if abs(kl_mid - kl_target) <= tol:
                        solved_eta = mid
                        break
                    # KL decreases with eta, so if kl_mid > target -> need larger eta
                    if kl_mid > kl_target:
                        lo = mid
                    else:
                        hi = mid
                    solved_eta = mid
            # clamp solved eta
            solved_eta = float(np.clip(solved_eta, eta_bounds[0], eta_bounds[1]))

        # compute final q_dist and kl using solved_eta
        q_dist, kl_np = compute_q_dist_and_kl(solved_eta)

    return actions.detach(), q_dist, kl_np, solved_eta


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


def evaluate_policy(policy: GaussianPolicy, env, device, n_eval_episodes: int = 5):
    """
    Run the policy for n_eval_episodes (stochastic sampling) and return list of episode returns.
    """
    returns = []
    for _ in range(n_eval_episodes):
        obs = torch.tensor(
            env.reset()[0], dtype=torch.float32, device=device
        ).unsqueeze(0)
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                action_tensor, _ = policy.sample(obs)
                action = action_tensor.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(
                0
            )
        returns.append(ep_ret)
    return returns


def main():
    args = argparse.ArgumentParser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # MPO / training hyper-parameters (descriptive names matching MPO paper)
    BATCH_SIZE = 64  # B: mini-batch size used for sampled updates
    NUM_TRAINING_EPISODES = 1000  # total episodes to run
    NUM_CANDIDATE_ACTIONS = 4  # K: candidate actions sampled per state in E-step
    MIN_REPLAY_SIZE = 100  # minimum number of stored transitions before training
    NUM_OPTIMIZATION_STEPS = (
        1000  # number of optimization (gradient) steps run when training is triggered
    )
    seed = 42
    q_lr = 0.0005
    pi_lr = 0.0005
    tau = 0.005
    dual_lr = 1e-3
    eta = 1.0
    kl_epsilon = 0.1
    policy_old_sync_frequency = 50  # frequency (in env steps) to sync pi_old <- pi
    log_dir = "./logs/mpo_experiment"
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Evaluation config
    eval_freq = 10  # run evaluations every `eval_freq` training episodes
    eval_episodes = 5  # number of eval episodes per evaluation

    wandb.init(
        project="mpo_project",
        config={
            "batch_size": BATCH_SIZE,
            "num_training_episodes": NUM_TRAINING_EPISODES,
            "num_candidate_actions": NUM_CANDIDATE_ACTIONS,
            "min_replay_size": MIN_REPLAY_SIZE,
            "num_optimization_steps": NUM_OPTIMIZATION_STEPS,
            "q_lr": q_lr,
            "pi_lr": pi_lr,
            "tau": tau,
            "dual_lr": dual_lr,
            "kl_epsilon": kl_epsilon,
            "seed": seed,
            "eval_freq": eval_freq,
            "eval_episodes": eval_episodes,
        },
        sync_tensorboard=True,
    )
    writer = SummaryWriter(log_dir)

    env = gymnasium.make("HalfCheetah-v5")
    env.reset(seed=seed)

    # create a separate eval env to avoid interfering with training env state
    eval_env = gymnasium.make("HalfCheetah-v5")
    eval_env.reset(seed=seed + 1007)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    q = QNetwork(obs_dim, act_dim).to(device)
    q_target = QNetwork(obs_dim, act_dim).to(device)
    q_target.load_state_dict(q.state_dict())

    pi = GaussianPolicy(obs_dim, act_dim, action_low, action_high).to(device)
    pi_old = GaussianPolicy(obs_dim, act_dim, action_low, action_high).to(device)
    pi_old.load_state_dict(pi.state_dict())

    q_optimizer = torch.optim.Adam(q.parameters(), lr=q_lr)
    pi_optimizer = torch.optim.Adam(pi.parameters(), lr=pi_lr)

    replay_buffer = NStepReplayBuffer(
        capacity=100000,
        obs_shape=env.observation_space.shape,
        act_shape=env.action_space.shape,
        n_step=5,
        gamma=0.99,
        device=device,
    )
    global_step = 0
    for episode in range(NUM_TRAINING_EPISODES):
        print("[Train] Starting episode %d ..." % (episode + 1))
        writer.add_scalar("info/episode", episode, global_step)
        obs = torch.tensor(
            env.reset()[0], dtype=torch.float32, device=device
        ).unsqueeze(0)
        done = False
        ep_return = 0.0
        # start timer for this episode
        start_time = time.time()
        while not done:
            global_step += 1
            action = None
            with torch.no_grad():
                action_tensor, logp_mu = pi_old.sample(obs)
                action = action_tensor.cpu().numpy()[0]
            next_states, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
            replay_buffer.push(
                obs.cpu().numpy()[0],
                action,
                reward,
                done,
                logp_mu.item(),
                next_states,
            )
            obs = torch.tensor(
                next_states, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # Trigger training when replay has enough data
            if len(replay_buffer) >= MIN_REPLAY_SIZE:
                print("[Train] Replay buffer batch of %d ..." % (MIN_REPLAY_SIZE))
                # Run a block of optimization steps (each step samples a mini-batch)
                for opt_iter in range(NUM_OPTIMIZATION_STEPS):
                    # Sample a mini-batch B of N (s, a, r) pairs from replay
                    obs_seq, act_seq, rew_seq, done_seq, next_obs, logp_mu_seq, seq_lens = (
                        replay_buffer.sample(BATCH_SIZE)
                    )

                    with torch.no_grad():
                        # [B, T, D_act] -> [B*T, D_act]
                        flat_obs_seq = obs_seq.view(-1, obs_dim)
                        flat_act_seq = act_seq.view(-1, act_dim)
                        flat_logp_pi_seq = pi.log_prob(flat_obs_seq, flat_act_seq)
                        logp_pi_seq = flat_logp_pi_seq.view(BATCH_SIZE, -1)

                        rho = torch.exp(logp_pi_seq - logp_mu_seq)
                        c = torch.clamp(rho, max=1.0)

                        # bootstrap from last next_obs
                        next_a, _ = pi.sample(next_obs)
                        q_next = q_target(next_obs, next_a)
                        target_q = q_next

                        # iterate backwards through sequence
                        for t in reversed(range(replay_buffer.n_step)):
                            rt = rew_seq[:, t]
                            ct = c[:, t]
                            donet = done_seq[:, t]

                            # get Q(s_t, a_t)
                            qt = q_target(obs_seq[:,t,:], act_seq[:,t,:])
                            # get V(s_t) by sampling K actions
                            with torch.no_grad():
                                next_actions_k, _ = pi.sample(obs_seq[:,t,:].unsqueeze(1).expand(-1, K, -1).reshape(B*K, obs_dim))
                                q_next_k = q_target(obs_seq[:,t,:].unsqueeze(1).expand(-1, K, -1).reshape(B*K, obs_dim), next_actions_k)
                                q_next_k = q_next_k.view(B, K, -1)
                                vt = q_next_k.mean(dim=1)

                            target_q = rt + 0.99 * (1.0 - donet) * (ct * (target_q - qt) + vt).squeeze()


                    q_sa = q(obs_seq[:,0,:], act_seq[:,0,:])
                    loss_q = F.mse_loss(q_sa, target_q)

                    q_optimizer.zero_grad()
                    loss_q.backward()
                    q_optimizer.step()

                    for tp, p in zip(q_target.parameters(), q.parameters()):
                        tp.data.mul_(1.0 - tau).add_(tau * p.data)

                    # Use E-step that can solve for eta (dual) robustly
                    actions_e, q_dist, kl_np, eta = policy_evaluation_e_step(
                        obs_seq[:,0,:],
                        pi_old,
                        q,
                        NUM_CANDIDATE_ACTIONS,
                        eta=eta,
                        solve_dual=True,
                        kl_target=kl_epsilon,
                        eta_bounds=(1e-8, 1e6),
                        max_iters=50,
                        tol=1e-4,
                    )

                    pi_loss = policy_evaluation_m_step(
                        pi, obs_seq[:,0,:], actions_e, q_dist, entropy_coeff=1e-3
                    )

                    pi_optimizer.zero_grad()
                    pi_loss.backward()
                    pi_optimizer.step()

                writer.add_scalar("loss/loss_q", loss_q.item(), global_step)
                writer.add_scalar("info/kl_np", kl_np, global_step)
                writer.add_scalar("info/eta", eta, global_step)
                writer.add_scalar("loss/pi_loss", pi_loss.item(), global_step)
            if policy_old_sync_frequency > 0:
                if global_step % policy_old_sync_frequency == 0:
                    pi_old.load_state_dict(pi.state_dict())
                    writer.add_scalar("info/policy_old_synced", 1.0, global_step)
                else:
                    writer.add_scalar("info/policy_old_synced", 0.0, global_step)

        episode_duration = time.time() - start_time
        writer.add_scalar("time/episode_duration", episode_duration, global_step)
        writer.add_scalar("rewards/ep_return", ep_return, global_step)
        print(
            f"[Train] episode={episode+1} global_step={global_step} ep_return={ep_return:.3f} duration={episode_duration:.3f}s"
        )

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
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode+1}.pt")
            torch.save(checkpoint, ckpt_path)
            # overwrite latest for convenience
            torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint_latest.pt"))
            writer.add_text("info/checkpoint_saved", ckpt_path, global_step)
        except Exception as e:
            # keep training even if checkpointing fails
            print(f"[Checkpoint] failed to save checkpoint: {e}")

        # Periodic evaluation: log to tensorboard, wandb, and console
        if (episode + 1) % eval_freq == 0:
            eval_returns = evaluate_policy(
                pi, eval_env, device, n_eval_episodes=eval_episodes
            )
            eval_mean = float(np.mean(eval_returns))
            writer.add_scalar("eval/ep_return_mean", eval_mean, global_step)
            print(
                f"[Eval] episode={episode+1} global_step={global_step} "
                f"eval_mean={eval_mean:.3f} eval_returns={[round(r,2) for r in eval_returns]}"
            )

    writer.close()
    wandb.finish()


if __name__ == "__main__":
    main()
