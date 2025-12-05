import os

import torch

from gaussian_policy import GaussianPolicy
from q_network import QNetwork


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


def checkpoint(
    checkpoint_dir: str,
    episode: int,
    global_step: int,
    q: QNetwork,
    q_target: QNetwork,
    pi: GaussianPolicy,
    pi_old: GaussianPolicy,
    q_optimizer: torch.optim.Optimizer,
    pi_optimizer: torch.optim.Optimizer,
):
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
    except Exception as e:
        # keep training even if checkpointing fails
        print(f"[Checkpoint] failed to save checkpoint: {e}")
