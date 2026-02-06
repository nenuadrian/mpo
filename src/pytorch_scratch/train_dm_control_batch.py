"""V-MPO (On-Policy Maximum a Posteriori Policy Optimisation) for dm_control."""

import argparse
import copy
import random

import gymnasium as gym
import numpy as np
import shimmy  # noqa: F401 (needed for dm_control Gymnasium registration)
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best-effort determinism (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dm_control_env(domain, task, seed=None):
    """Create a dm_control environment via shimmy's Gymnasium wrapper."""
    env = gym.make(
        f"dm_control/{domain}-{task}-v0",
    )
    if seed is not None:
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
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
    """Compute GAE-λ advantages and corresponding value-function targets."""
    T = len(rewards)
    advantages = torch.zeros(T, device=values.device)
    gae = 0.0

    # Walk backwards through the rollout to accumulate the
    # exponentially-weighted sum of TD errors.
    for t in reversed(range(T)):
        # 1-step TD error: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        # (1 - done) masks the bootstrap when an episode ended at step t.
        delta = rewards[t] + gamma * dones[t] * values[t + 1] - values[t]
        # Recursive GAE accumulation: A_t = δ_t + γλ · A_{t+1}
        gae = delta + gamma * lam * dones[t] * gae
        advantages[t] = gae

    # Value-function regression targets: R_t = A_t + V(s_t)
    returns = advantages + values[:-1]
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
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        mean = self.net(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def dist(self, x):
        mean, std = self.forward(x)
        return torch.distributions.Normal(mean, std)

    def sample(self, x):
        return self.dist(x).sample()


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


# VMPO Trainer
# Orchestrates the full VMPO training loop:
#   - Maintains two copies of the policy: π_θ (updated) and π_old (behaviour).
#   - Maintains a learned temperature η (stored in log-space for positivity).
#   - Each iteration: collect → GAE → E-step → M-step → dual η update →
#     value update → sync π_old ← π_θ.
class DMControlBatchTrainer:
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
        seed: int = 42,
    ):
        set_seed(seed)
        self.env = make_dm_control_env(domain, task, seed=seed)

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
        self.opt_eta = optim.Adam([self.eta], lr=1e-3)

        # Rollout / GAE hyper-parameters
        self.rollout_steps = rollout_steps
        self.gamma = gamma
        self.lam = lam
        # ε_η  – the KL bound on the E-step weight distribution.
        # Smaller → more uniform weights → more conservative updates.
        self.n_temperature_epsilon = n_temperature_epsilon

        # Parametric KL multiplier α (stored in log-space for positivity)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(5.0), device=device))
        self.opt_alpha = optim.Adam([self.log_alpha], lr=1e-3)

        self.eps_alpha = 0.01

        # Running episode statistics (accumulated across rollout boundaries)
        self.ep_return = 0.0
        self.ep_length = 0
        self.total_env_steps = 0

        self.domain = domain
        self.task = task

    # Data Collection  (rollout with π_old)

    def collect(self):
        """Collect a fixed-length rollout using the behaviour policy π_old.

        Returns tensors of observations, actions, rewards, done flags,
        value estimates (T+1 entries – includes bootstrap), and a list of
        completed episode returns for logging.
        """
        obs, acts, rews, dones, vals = [], [], [], [], []

        s, _ = self.env.reset()
        s = flatten_obs(s)

        episode_returns = []  # returns of episodes that completed during this rollout

        for _ in range(self.rollout_steps):
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)

            # Sample action from the *behaviour* policy π_old (no grad needed)
            # and record the value estimate V(s_t) for GAE.
            with torch.no_grad():
                a = self.policy_old.sample(s_t).cpu().numpy().squeeze(0)
                a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
                v = self.value(s_t).item()

            s2, r, terminated, truncated, _ = self.env.step(a)
            s2 = flatten_obs(s2)
            self.total_env_steps += 1
            done = 0 if terminated else 1

            # Accumulate episode-level statistics
            self.ep_return += r
            self.ep_length += 1

            obs.append(s)
            acts.append(a)
            rews.append(r)
            dones.append(float(done))
            vals.append(v)

            if done or truncated:
                episode_returns.append(self.ep_return)
                self.ep_return = 0.0
                self.ep_length = 0
                s, _ = self.env.reset()
                s = flatten_obs(s)
            else:
                s = s2

        # Bootstrap value for the last state (needed by GAE to compute the
        # advantage of the final transition in the rollout).
        with torch.no_grad():
            v_last = self.value(
                torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
            ).item()

        vals.append(v_last)  # vals has T+1 entries

        return (
            torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(acts, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(rews, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(dones, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(vals, dtype=np.float32)).to(device),
            episode_returns,
        )

    def train_once(self):
        """Execute one full VMPO iteration:
        collect → GAE → E-step → M-step → dual η update → value update.
        """

        # Data Collection  –  rollout with behaviour policy π_old
        s, a, r, d, v, episode_returns = self.collect()

        # Advantage Estimation (GAE-λ)
        # Compute advantages A_t and value-regression targets R_t using
        # the value network's predictions from the rollout.
        adv, ret = compute_gae(r, v, d, self.gamma, self.lam)

        # Advantage Pre-processing (Top-K Masking)
        # We only want to weight the "good" half of the samples.
        # This prevents the exponential weights from being dominated by outliers
        # and acts as a trust-region filter.
        with torch.no_grad():
            # Calculate top 50% threshold
            top_k_threshold = torch.quantile(adv, 0.5)
            # Create a boolean mask of size (T,)
            mask = adv >= top_k_threshold

            # Select only top-k advantages
            adv_selected = adv[mask]

            # Centre advantages for numerical stability of softmax
            # (Note: V-MPO formulation is invariant to shifting A, but float precision isn't)
            adv_selected = adv_selected - adv_selected.mean()
            adv = adv - adv.mean()
            mask = adv >= adv_selected.median()
            adv_selected = adv[mask]

            # E-STEP: Compute weights on the subset
            eta = self.eta.exp()
            # w_i = exp(A_i / eta) / Z
            q_weights = torch.softmax(adv_selected / eta, dim=0)

        # M-STEP & Value Update Loop
        # We iterate multiple times over the batch to fully regress the policy onto the target q.

        n_epochs = 4

        # Prepare datasets:
        # Policy uses ONLY Top-K data (masked)
        s_top = s[mask]
        a_top = a[mask]
        q_top = q_weights

        # Value function uses ALL data (unmasked)
        s_full = s
        ret_full = ret

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0

        for _ in range(n_epochs):
            # --- Policy Update Loop (on Top-K data) ---
            idx = torch.randperm(len(s_top))
            mb_s = s_top[idx]
            mb_a = a_top[idx]
            mb_w = q_top[idx]

            # M-STEP: Weighted Maximum Likelihood
            # New policy
            new_dist = self.policy.dist(mb_s)
            logp = new_dist.log_prob(mb_a).sum(dim=-1)

            # Old policy (detached)
            with torch.no_grad():
                old_dist = self.policy_old.dist(mb_s)

            # L_pi = - sum( w_i * log pi(a|s) )
            # Note: mb_w sum is not 1 here due to mini-batching, but gradient scales linearly.
            # Since weights sum to 1 over the full set, we sum here (not mean).
            policy_loss = -(mb_w * logp).sum()

            # KL(π_old || π_new)
            kl = (
                torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1).mean()
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

            # --- Value Update Loop (on ALL data) ---
            # Value function learns from all transitions to properly estimate V(s)
            idx = torch.randperm(len(s_full))
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

        # DUAL η UPDATE  –  E-step temperature optimisation
        # We perform this ONCE per rollout using the cached `adv_selected`.
        # Re-calculate eta from parameter to ensure the graph is connected.

        eta = self.eta.exp()
        T_factor = torch.tensor(float(len(adv_selected)), device=adv.device)

        # The dual objective: η * ε + η * log( mean( exp(A/η) ) )
        # Implemented using logsumexp for stability:
        # log( mean( exp(A/η) ) ) = log( sum(exp(A/η)) / N ) = logsumexp(A/η) - log(N)
        eta_loss = eta * (
            self.n_temperature_epsilon
            + torch.logsumexp(adv_selected.detach() / eta, dim=0)
            - torch.log(T_factor)
        )

        self.opt_eta.zero_grad()
        eta_loss.backward()
        self.opt_eta.step()

        # Diagnostics / Logging
        with torch.no_grad():
            new_dist = self.policy.dist(s_full[:512])
            entropy = new_dist.entropy().sum(dim=-1).mean()

        metrics = {
            "entropy": entropy.item(),
            "rollout_return": float(np.sum(episode_returns))
            / max(1, len(episode_returns)),
            "policy_loss": total_policy_loss / n_epochs,  # M-step loss
            "value_loss": total_value_loss / n_epochs,  # critic MSE
            "eta": eta.item(),  # current temperature
            "eta_loss": eta_loss.item(),  # dual objective value
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
    parser = argparse.ArgumentParser(
        description="Train dm_control with VMPO (minibatch)"
    )
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    wandb.init(
        project="dm-control-vmpo",
        config=vars(args),
        group=f"{args.domain}-{args.task}",
        name=f"{args.domain}-{args.task}-vmpo-batch",
    )

    DMControlBatchTrainer(
        domain=args.domain,
        task=args.task,
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        lam=args.lam,
        lr=args.lr,
        n_temperature_epsilon=args.n_temperature_epsilon,
        eta_initial=args.eta_initial,
        seed=args.seed,
    ).train(iters=args.iters)
    wandb.finish()
