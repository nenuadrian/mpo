# V-MPO (On-Policy Maximum a Posteriori Policy Optimisation) for dm_control

import numpy as np
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import shimmy
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dm_control_env(domain, task, seed=None):
    """Create a dm_control environment via shimmy's Gymnasium wrapper.

    Args:
        domain: dm_control domain name (e.g. 'cheetah', 'walker', 'cartpole').
        task:   dm_control task name (e.g. 'run', 'walk', 'swingup').
        seed:   optional random seed.

    Returns:
        A Gymnasium-compatible environment with continuous observation and
        action spaces.
    """
    env = gym.make(
        f"dm_control/{domain}-{task}-v0",
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def flatten_obs(obs):
    """Flatten a dm_control observation dict/OrderedDict into a 1-D numpy array."""
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs.keys()):
            parts.append(np.asarray(obs[key], dtype=np.float32).flatten())
        return np.concatenate(parts)
    return np.asarray(obs, dtype=np.float32).flatten()


# Generalised Advantage Estimation (GAE)
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute GAE-λ advantages and corresponding value-function targets.

    GAE provides a bias–variance trade-off for advantage estimation:
      A_t^{GAE} = Σ_{l=0}^{T-t-1} (γλ)^l · δ_{t+l}
    where δ_t = r_t + γ V(s_{t+1}) - V(s_t)  is the 1-step TD error.

    Args:
        rewards : (T,)   tensor of rewards.
        values  : (T+1,) tensor of value estimates (includes bootstrap V(s_T)).
        dones   : (T,)   tensor of done flags (1.0 = episode boundary).
        gamma   : discount factor.
        lam     : GAE λ (0 → pure TD, 1 → pure MC).

    Returns:
        advantages : (T,) GAE advantages.
        returns    : (T,) targets for value regression  (A_t + V(s_t)).
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=values.device)
    gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * (1 - dones[t]) * values[t + 1] - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


# Policy Network  (π_θ)
# Outputs a Gaussian distribution over the continuous action space.
# Architecture: MLP with two hidden layers (256 units each).
# Outputs mean and a state-independent log_std parameter.
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),  # mean of Gaussian
        )
        # State-independent log standard deviation
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        """Return (mean, std) for each batch element."""
        mean = self.net(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def dist(self, x):
        """Return a Normal distribution for the given observations."""
        mean, std = self.forward(x)
        return torch.distributions.Normal(mean, std)

    def sample(self, x):
        """Sample an action and return it (no gradient)."""
        d = self.dist(x)
        return d.sample()

    def log_prob(self, x, a):
        """Compute log π_θ(a | s) for given state-action pairs.
        Returns per-sample log-prob (sum over action dimensions)."""
        d = self.dist(x)
        return d.log_prob(a).sum(dim=-1)


# Value Network  V_φ(s)
# MLP with two hidden layers (256 units each), scalar output.
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
        """Return V(s) as a scalar per batch element."""
        return self.net(x).squeeze(-1)


# VMPO Trainer
class DMControlTrainer:
    def __init__(
        self,
        domain="cheetah",
        task="run",
        rollout_steps=2048,
        gamma=0.99,
        lam=0.95,
        lr=3e-4,
        n_temperature_epsilon=0.1,
        eta_initial=0.0,
    ):
        self.env = make_dm_control_env(domain, task)

        # Determine observation and action dimensions
        # dm_control obs may be a Dict space
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            obs_dim = sum(
                int(np.prod(v.shape))
                for v in self.env.observation_space.spaces.values()
            )
        else:
            obs_dim = int(np.prod(self.env.observation_space.shape))

        act_dim = int(np.prod(self.env.action_space.shape))
        self.act_low = torch.tensor(
            self.env.action_space.low, dtype=torch.float32, device=device
        )
        self.act_high = torch.tensor(
            self.env.action_space.high, dtype=torch.float32, device=device
        )

        # π_θ  – the policy we optimise (M-step target)
        self.policy = GaussianPolicy(obs_dim, act_dim).to(device)
        # π_old – frozen copy used as the behaviour policy
        self.policy_old = copy.deepcopy(self.policy).eval()
        # V_φ  – state-value function
        self.value = ValueNet(obs_dim).to(device)

        self.opt_pi = optim.Adam(self.policy.parameters(), lr=lr)
        self.opt_v = optim.Adam(self.value.parameters(), lr=lr)

        # η (eta) – the E-step temperature (log-space)
        self.eta = nn.Parameter(torch.tensor(eta_initial, device=device))
        self.opt_eta = optim.Adam([self.eta], lr=1e-3)

        self.rollout_steps = rollout_steps
        self.gamma = gamma
        self.lam = lam
        self.n_temperature_epsilon = n_temperature_epsilon

        # Parametric KL multiplier α (log-space)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(5.0), device=device))
        self.opt_alpha = optim.Adam([self.log_alpha], lr=1e-3)
        self.eps_alpha = 0.01

        self.ep_return = 0.0
        self.ep_length = 0
        self.total_env_steps = 0

    def collect(self):
        """Collect a fixed-length rollout using the behaviour policy π_old."""
        obs, acts, rews, dones, vals = [], [], [], [], []

        s, _ = self.env.reset()
        s = flatten_obs(s)

        episode_returns = []

        for _ in range(self.rollout_steps):
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                a = self.policy_old.sample(s_t).cpu().numpy().squeeze(0)
                # Clip action to valid range
                a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
                v = self.value(s_t).item()

            s2, r, terminated, truncated, _ = self.env.step(a)
            s2 = flatten_obs(s2)
            self.total_env_steps += 1
            done = terminated or truncated

            self.ep_return += r
            self.ep_length += 1

            obs.append(s)
            acts.append(a)
            rews.append(r)
            dones.append(float(done))
            vals.append(v)

            if done:
                episode_returns.append(self.ep_return)
                self.ep_return = 0.0
                self.ep_length = 0
                s, _ = self.env.reset()
                s = flatten_obs(s)
            else:
                s = s2

        with torch.no_grad():
            v_last = self.value(
                torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
            ).item()

        vals.append(v_last)

        return (
            torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(acts, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(rews, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(dones, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(vals, dtype=np.float32)).to(device),
            episode_returns,
        )

    def train_once(self):
        """Execute one full VMPO iteration."""

        s, a, r, d, v, episode_returns = self.collect()

        adv, ret = compute_gae(r, v, d, self.gamma, self.lam)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # M-STEP & Value Update
        s_top = s
        a_top = a
        w_top = torch.softmax(adv, dim=0).detach()

        s_full = s
        ret_full = ret

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0

        n_epochs = 1
        batch_size = len(s_top)
        for epoch in range(n_epochs):
            perm_top = torch.randperm(len(s_top))

            for start in range(0, len(s_top), batch_size):
                idx = perm_top[start : start + batch_size]
                mb_s = s_top[idx]
                mb_a = a_top[idx]
                mb_w = w_top[idx]

                # New policy
                new_dist = self.policy.dist(mb_s)
                logp = new_dist.log_prob(mb_a).sum(dim=-1)

                # Old policy (detached)
                with torch.no_grad():
                    old_dist = self.policy_old.dist(mb_s)

                # L_pi = - sum( w_i * log pi(a|s) )
                policy_loss = -(mb_w * logp).sum()

                # KL(π_old || π_new) — sum over action dims, mean over batch
                kl = (
                    torch.distributions.kl_divergence(old_dist, new_dist)
                    .sum(dim=-1)
                    .mean()
                )
                total_kl += kl.item()

                alpha = self.log_alpha.exp()

                # θ update
                self.opt_pi.zero_grad()
                (policy_loss + alpha.detach() * kl).backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.opt_pi.step()

                # α update
                self.opt_alpha.zero_grad()
                (alpha * (self.eps_alpha - kl.detach())).backward()
                self.opt_alpha.step()

                total_policy_loss += policy_loss.item()

            # --- Value Update Loop ---
            perm_full = torch.randperm(len(s_full))
            for start in range(0, len(s_full), batch_size):
                idx = perm_full[start : start + batch_size]
                mb_s = s_full[idx]
                mb_ret = ret_full[idx]

                values = self.value(mb_s)
                value_loss = ((values - mb_ret) ** 2).mean()

                self.opt_v.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.opt_v.step()
                total_value_loss += value_loss.item()

        self.policy_old.load_state_dict(self.policy.state_dict())

        # DUAL η UPDATE
        eta = self.eta.exp()
        T_factor = torch.tensor(float(len(adv)), device=adv.device)

        eta_loss = eta * (
            self.n_temperature_epsilon
            + torch.logsumexp(adv.detach() / eta, dim=0)
            - torch.log(T_factor)
        )

        self.opt_eta.zero_grad()
        eta_loss.backward()
        self.opt_eta.step()

        # Diagnostics
        with torch.no_grad():
            new_dist = self.policy.dist(s_full[:512])
            entropy = new_dist.entropy().sum(dim=-1).mean()

        metrics = {
            "entropy": entropy.item(),
            "rollout_return": float(np.sum(episode_returns))
            / max(1, len(episode_returns)),
            "policy_loss": total_policy_loss / n_epochs,
            "value_loss": total_value_loss / n_epochs,
            "eta": eta.item(),
            "eta_loss": eta_loss.item(),
            "alpha": self.log_alpha.exp().item(),
            "kl": total_kl / max(1, len(s_top)),
        }

        if episode_returns:
            metrics.update(
                {
                    "episode_return_mean": np.mean(episode_returns),
                    "episode_return_max": np.max(episode_returns),
                    "episode_return_min": np.min(episode_returns),
                    "episodes_in_rollout": len(episode_returns),
                }
            )

        return metrics

    def train(self, iters=10_000):
        """Run `iters` VMPO iterations, logging to W&B each step."""
        for it in range(iters):
            info = self.train_once()
            wandb.log(info, step=self.total_env_steps)
            if it % 10 == 0:
                print(it, info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dm_control with VMPO")
    parser.add_argument(
        "--domain",
        type=str,
        default="cheetah",
        help="dm_control domain name (e.g. cheetah, walker, cartpole, humanoid)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="run",
        help="dm_control task name (e.g. run, walk, swingup, stand)",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=2048,
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
        default=0.1,
        help="ε_η – KL bound on E-step weight distribution",
    )
    parser.add_argument(
        "--iters", type=int, default=10_000, help="Total number of VMPO iterations"
    )
    parser.add_argument(
        "--eta_initial",
        type=float,
        default=0.0,
        help="Initial value for log(η) temperature parameter",
    )
    args = parser.parse_args()

    wandb.init(project="dm-control-vmpo", config=vars(args))
    DMControlTrainer(
        domain=args.domain,
        task=args.task,
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        lam=args.lam,
        lr=args.lr,
        n_temperature_epsilon=args.n_temperature_epsilon,
        eta_initial=args.eta_initial,
    ).train(iters=args.iters)
    wandb.finish()
