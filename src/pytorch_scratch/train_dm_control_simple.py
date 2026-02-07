import argparse
import copy
import random

import gymnasium as gym  # type: ignore
import numpy as np  # type: ignore
import shimmy  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
import wandb
import os

from utils import set_seed, make_dm_control_env, evaluate, flatten_obs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_gae(rewards, values, next_values, bootstrap_masks, gamma=0.99, lam=0.95):
    """Compute GAE-λ advantages and corresponding value-function targets.

    bootstrap_masks[t] = 0 → terminated=True (true terminal): no bootstrap
    bootstrap_masks[t] = 1 → non-terminal or truncated=True (time-limit): bootstrap
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=values.device)
    gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * bootstrap_masks[t] * next_values[t] - values[t]
        gae = delta + gamma * lam * bootstrap_masks[t] * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous action spaces."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        # initialise conservatively
        self.log_std = nn.Parameter(torch.ones(act_dim) * -1.0)

    def forward(self, x):
        mean = self.net(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def dist(self, x):
        mean, std = self.forward(x)
        return torch.distributions.Normal(mean, std)

    def sample(self, x):
        return self.dist(x).sample()

    def act_deterministic(self, x):
        mean, _ = self.forward(x)
        return mean


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DMControlVmpoTrainer:
    def __init__(
        self,
        domain: str,
        task: str,
        rollout_steps: int,
        gamma: float,
        lam: float,
        lr: float,
        n_temperature_epsilon: float,
        eta_initial: float,
        seed: int,
        eps_alpha_mu: float,
        eps_alpha_sigma: float,
        n_value_updates: int,
        n_policy_updates: int,
        eta_lr: float,
    ):
        set_seed(seed)
        self.env = make_dm_control_env(domain, task, seed=seed)
        self.checkpoints_dir = f"checkpoints/{domain}_{task}"
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        if isinstance(self.env.observation_space, gym.spaces.Dict):
            obs_dim = sum(
                int(np.prod(v.shape))
                for v in self.env.observation_space.spaces.values()
            )
        else:
            obs_dim = int(np.prod(self.env.observation_space.shape))

        act_dim = int(np.prod(self.env.action_space.shape))

        self.policy = GaussianPolicy(obs_dim, act_dim).to(device)
        # π_old – frozen copy used as the behaviour policy for data collection
        #         and as the reference distribution for KL measurement.
        self.policy_old = copy.deepcopy(self.policy).eval()
        self.value = ValueNet(obs_dim).to(device)

        # Separate optimisers for policy, value, and dual variable η
        self.opt_pi = optim.Adam(self.policy.parameters(), lr=lr)
        self.opt_v = optim.Adam(self.value.parameters(), lr=lr)

        # η (eta) – the E-step temperature, stored in log-space so that
        # exp(self.eta) is always positive.  Optimised with its own Adam.
        self.eta = nn.Parameter(torch.tensor(eta_initial, device=device))
        self.opt_eta = optim.Adam([self.eta], lr=eta_lr)

        # Rollout / GAE hyper-parameters
        self.rollout_steps = rollout_steps
        self.gamma = gamma
        self.lam = lam
        # ε_η  – the KL bound on the E-step weight distribution.
        # Smaller → more uniform weights → more conservative updates.
        self.n_temperature_epsilon = n_temperature_epsilon

        # Parametric KL multiplier α (stored in log-space for positivity)
        self.log_alpha_mu = nn.Parameter(torch.tensor(np.log(5.0), device=device))
        self.log_alpha_sigma = nn.Parameter(torch.tensor(np.log(1.0), device=device))

        self.opt_alpha_mu = optim.Adam([self.log_alpha_mu], lr=lr)
        self.opt_alpha_sigma = optim.Adam([self.log_alpha_sigma], lr=lr)

        # Separate trust-region constraints
        self.eps_alpha_mu = eps_alpha_mu
        self.eps_alpha_sigma = eps_alpha_sigma

        # Running episode statistics (accumulated across rollout boundaries)
        self.ep_return = 0.0
        self.ep_length = 0
        self.total_env_steps = 0

        self.n_value_updates = n_value_updates
        self.n_policy_updates = n_policy_updates

        self.domain = domain
        self.task = task

    def collect(self):
        obs, acts, rews, bootstrap_masks, vals, next_vals = [], [], [], [], [], []

        s, _ = self.env.reset()
        s = flatten_obs(s)

        episode_returns = []

        for _ in range(self.rollout_steps):
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                a = self.policy_old.sample(s_t).cpu().numpy().squeeze(0)
                a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
                v = self.value(s_t).item()

            s2, r, terminated, truncated, _ = self.env.step(a)
            s2 = flatten_obs(s2)
            self.total_env_steps += 1

            # Gymnasium gives both for a reason:
            # - terminated => true terminal => do NOT bootstrap
            # - truncated  => time limit    => DO bootstrap, but still reset
            bootstrap_mask = 0.0 if terminated else 1.0

            with torch.no_grad():
                v_next = self.value(
                    torch.tensor(s2, dtype=torch.float32).unsqueeze(0).to(device)
                ).item()

            self.ep_return += r
            self.ep_length += 1

            obs.append(s)
            acts.append(a)
            rews.append(r)
            bootstrap_masks.append(bootstrap_mask)
            vals.append(v)
            next_vals.append(v_next)

            if terminated or truncated:
                episode_returns.append(self.ep_return)
                self.ep_return = 0.0
                self.ep_length = 0
                s, _ = self.env.reset()
                s = flatten_obs(s)
            else:
                s = s2

        return (
            torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(acts, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(rews, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(bootstrap_masks, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(vals, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(next_vals, dtype=np.float32)).to(device),
            episode_returns,
        )

    def train_once(self):

        # Data Collection  –  rollout with behaviour policy π_old
        s, a, r, bootstrap_masks, v, v_next, episode_returns = self.collect()

        # Advantage Estimation (GAE-λ)
        adv, ret = compute_gae(r, v, v_next, bootstrap_masks, self.gamma, self.lam)

        # Advantage Pre-processing (Top-K Masking)
        # We only want to weight the "good" half of the samples.
        # This prevents the exponential weights from being dominated by outliers
        # and acts as a trust-region filter.

        # 3. Centre for numerical stability ONLY
        adv = adv - adv.mean()

        # 4. E-step weights
        eta = self.eta.exp()
        with torch.no_grad():
            q_weights = torch.softmax(adv / eta, dim=0)
        ess = 1.0 / torch.sum(q_weights**2)
        ess_frac = ess / q_weights.numel()

        # M-STEP & Value Update Loop
        # We iterate multiple times over the batch to fully regress the policy onto the target q.

        # Prepare datasets:
        # Policy uses ONLY Top-K data (masked)
        s_top = s
        a_top = a

        # Value function uses ALL data (unmasked)

        total_policy_loss = 0.0
        total_kl_mu = 0.0
        total_kl_sigma = 0.0

        for _ in range(self.n_policy_updates):
            # --- Policy Update Loop (on Top-K data) ---

            # M-STEP: Weighted Maximum Likelihood
            # (A) Policy loss — top-k batch
            new_dist_top = self.policy.dist(s_top)
            logp = new_dist_top.log_prob(a_top).sum(dim=-1)
            policy_loss = -(q_weights * logp).sum()

            # (B) KL constraint — full batch
            with torch.no_grad():
                mu_old, std_old = self.policy_old(s)
            mu_old = mu_old.detach()
            std_old = std_old.detach()
            log_std_old = std_old.log()

            mu_new, std_new = self.policy(s)
            log_std_new = std_new.log()

            # Mean KL (behavioural shift only; uses old variance)
            kl_mu = 0.5 * (((mu_new - mu_old) ** 2) / (std_old**2)).sum(dim=-1).mean()

            # Variance KL (shape change only)
            kl_sigma = (
                0.5
                * (
                    (std_old**2) / (std_new**2)
                    - 1.0
                    + 2.0 * (log_std_new - log_std_old)
                )
                .sum(dim=-1)
                .mean()
            )

            total_kl_mu += kl_mu.item()
            total_kl_sigma += kl_sigma.item()

            alpha_mu = self.log_alpha_mu.exp()
            alpha_sigma = self.log_alpha_sigma.exp()

            # θ update
            self.opt_pi.zero_grad()
            (
                policy_loss
                + alpha_mu.detach() * kl_mu
                + alpha_sigma.detach() * kl_sigma
            ).backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.opt_pi.step()

            # hard clamp σ every update
            with torch.no_grad():
                self.policy.log_std.clamp_(-5.0, 1.0)

            # α updates (dual ascent)
            self.opt_alpha_mu.zero_grad()
            (alpha_mu * (self.eps_alpha_mu - kl_mu.detach())).backward()
            self.opt_alpha_mu.step()

            self.opt_alpha_sigma.zero_grad()
            (alpha_sigma * (self.eps_alpha_sigma - kl_sigma.detach())).backward()
            self.opt_alpha_sigma.step()

            # Projection: enforce α >= 1e-8 (in log-space)
            with torch.no_grad():
                self.log_alpha_mu.clamp_(min=np.log(1e-8))
                self.log_alpha_sigma.clamp_(min=np.log(1e-8))

            total_policy_loss += policy_loss.item()

        # --- Value Update Loop (on ALL data) ---
        # Value function learns from all transitions to properly estimate V(s)

        total_value_loss = 0.0

        for _ in range(self.n_value_updates):
            values = self.value(s)
            value_loss = ((values - ret) ** 2).mean()

            self.opt_v.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.opt_v.step()

            total_value_loss += value_loss.item()

        self.policy_old.load_state_dict(self.policy.state_dict())

        # DUAL η UPDATE  –  E-step temperature optimisation
        # We perform this ONCE per rollout using the cached `adv`.
        # Re-calculate eta from parameter to ensure the graph is connected.

        eta = self.eta.exp()
        log_batch_size = torch.tensor(float(len(adv)), device=adv.device)

        # The dual objective: η * ε + η * log( mean( exp(A/η) ) )
        # Implemented using logsumexp for stability:
        # log( mean( exp(A/η) ) ) = log( sum(exp(A/η)) / N ) = logsumexp(A/η) - log(N)
        eta_loss = eta * (
            self.n_temperature_epsilon
            + torch.logsumexp(adv.detach() / eta, dim=0)
            - torch.log(log_batch_size)
        )

        self.opt_eta.zero_grad()
        eta_loss.backward()
        self.opt_eta.step()

        # Projection: enforce η >= 1e-8 (in log-space)
        with torch.no_grad():
            self.eta.clamp_(min=np.log(1e-8))

        # Recompute η for logging after projection/update
        eta = self.eta.exp()

        # Diagnostics / Logging
        with torch.no_grad():
            new_dist = self.policy.dist(s)
            entropy = new_dist.entropy().sum(dim=-1).mean()

        metrics = {
            "train/entropy": entropy.item(),
            "train/rollout_return": float(np.sum(episode_returns))
            / max(1, len(episode_returns)),
            "train/policy_loss": total_policy_loss
            / self.n_policy_updates,  # M-step loss
            "train/value_loss": total_value_loss / self.n_value_updates,  # critic MSE
            "train/eta": eta.item(),  # current temperature
            "train/eta_loss": eta_loss.item(),  # dual objective value
            "train/alpha_mu": self.log_alpha_mu.exp().item(),
            "train/alpha_sigma": self.log_alpha_sigma.exp().item(),
            "train/kl_mu": total_kl_mu / self.n_policy_updates,
            "train/kl_sigma": total_kl_sigma / self.n_policy_updates,
            "train/kl": (total_kl_mu + total_kl_sigma) / self.n_policy_updates,
            "train/ess": ess.item(),
            "train/ess_frac": ess_frac.item(),
            "train/ess_ratio": ess.item() / adv.numel(),
        }

        if episode_returns:
            metrics.update(
                {
                    "train/episode_return_mean": np.mean(episode_returns),
                    "train/episode_return_max": np.max(episode_returns),
                    "train/episode_return_min": np.min(episode_returns),
                    "train/episodes_in_rollout": len(episode_returns),
                }
            )

        return metrics

    def train(self, iters=10_000):
        for it in range(iters):
            info = self.train_once()
            if it % 30 == 0:
                eval_metrics = evaluate(
                    device, self.policy, self.domain, self.task, n_episodes=10
                )
                info.update(eval_metrics)
            wandb.log(info, step=self.total_env_steps)  # type: ignore
            if it % 10 == 0:
                print(it, info)
            if it % 30 == 0:
                self.save_checkpoint(f"{self.checkpoints_dir}/ckpt.pt", it)

    def save_checkpoint(self, path, it):
        ckpt = {
            "iteration": it,
            "policy": self.policy.state_dict(),
            "policy_old": self.policy_old.state_dict(),
            "value": self.value.state_dict(),
            "opt_pi": self.opt_pi.state_dict(),
            "opt_v": self.opt_v.state_dict(),
            "opt_eta": self.opt_eta.state_dict(),
            "opt_alpha_mu": self.opt_alpha_mu.state_dict(),
            "opt_alpha_sigma": self.opt_alpha_sigma.state_dict(),
            "eta": self.eta.detach().cpu(),
            "log_alpha_mu": self.log_alpha_mu.detach().cpu(),
            "log_alpha_sigma": self.log_alpha_sigma.detach().cpu(),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        torch.save(ckpt, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        type=str,
    )
    parser.add_argument(
        "--task",
        type=str,
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=4096,
        help="Number of environment steps per rollout",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor γ")
    parser.add_argument(
        "--lam", type=float, default=0.95, help="GAE λ (bias-variance trade-off)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for policy and value networks",
    )
    parser.add_argument(
        "--n_temperature_epsilon",
        type=float,
        default=0.5,
        help="ε_η – KL bound on E-step weight distribution",
    )
    parser.add_argument(
        "--iters", type=int, default=10_000, help="Total number of VMPO iterations"
    )
    parser.add_argument(
        "--eta_initial",
        type=float,
        default=-1.0,
        help="Initial value for log(η) temperature parameter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--eps_alpha_mu",
        type=float,
        default=1.0,
        help="ε_α for mean KL constraint (smaller → more conservative)",
    )
    parser.add_argument(
        "--eps_alpha_sigma",
        type=float,
        default=0.1,
        help="ε_α for variance KL constraint (smaller → more conservative)",
    )
    parser.add_argument(
        "--n_value_updates",
        type=int,
        default=2,
        help="Number of value function updates per iteration",
    )
    parser.add_argument(
        "--n_policy_updates",
        type=int,
        default=4,
        help="Number of policy updates per iteration",
    )
    parser.add_argument(
        "--eta_lr",
        type=float,
        default=1e-4,
        help="Learning rate for dual variable η",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    wandb.init(  # type: ignore
        project="dm-control-vmpo",
        config=vars(args),
        group=f"{args.domain}-{args.task}",
        name=f"{args.domain}-{args.task}-vmpo-eta{args.eta_initial}-eps{args.n_temperature_epsilon}-seed{args.seed}",
    )

    DMControlVmpoTrainer(
        domain=args.domain,
        task=args.task,
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        lam=args.lam,
        lr=args.lr,
        n_temperature_epsilon=args.n_temperature_epsilon,
        eta_initial=args.eta_initial,
        seed=args.seed,
        eps_alpha_mu=args.eps_alpha_mu,
        eps_alpha_sigma=args.eps_alpha_sigma,
        n_value_updates=args.n_value_updates,
        n_policy_updates=args.n_policy_updates,
        eta_lr=args.eta_lr,
    ).train(iters=args.iters)
    wandb.finish()  # type: ignore
