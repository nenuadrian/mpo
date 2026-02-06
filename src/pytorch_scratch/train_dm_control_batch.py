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
        rollout_steps=4096,
        gamma=0.99,
        lam=0.95,
        lr=3e-4,
        n_temperature_epsilon=0.1,
        eta_initial=0.0,
        seed: int = 42,
        eps_alpha_mu: float = 0.5,
        eps_alpha_sigma: float = 0.01,
        top_k_fraction: float = 1.0,
        n_value_updates: int = 2,
        n_policy_updates: int = 4,
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
        self.log_alpha_mu = nn.Parameter(torch.tensor(np.log(5.0), device=device))
        self.log_alpha_sigma = nn.Parameter(torch.tensor(np.log(1.0), device=device))

        self.opt_alpha_mu = optim.Adam([self.log_alpha_mu], lr=1e-3)
        self.opt_alpha_sigma = optim.Adam([self.log_alpha_sigma], lr=1e-3)

        # Separate trust-region constraints
        self.eps_alpha_mu = eps_alpha_mu
        self.eps_alpha_sigma = eps_alpha_sigma

        # Running episode statistics (accumulated across rollout boundaries)
        self.ep_return = 0.0
        self.ep_length = 0
        self.total_env_steps = 0
        self.top_k_fraction = top_k_fraction

        self.n_value_updates = n_value_updates
        self.n_policy_updates = n_policy_updates

        self.domain = domain
        self.task = task

    # Data Collection  (rollout with π_old)

    def collect(self):
        """Collect a fixed-length rollout using the behaviour policy π_old.

        Returns tensors of observations, actions, rewards, bootstrap masks,
        value estimates V(s_t), next value estimates V(s_{t+1}), and a list of
        completed episode returns for logging.
        """
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
        """Execute one full VMPO iteration:
        collect → GAE → E-step → M-step → dual η update → value update.
        """

        # Data Collection  –  rollout with behaviour policy π_old
        s, a, r, bootstrap_masks, v, v_next, episode_returns = self.collect()

        # Advantage Estimation (GAE-λ)
        adv, ret = compute_gae(r, v, v_next, bootstrap_masks, self.gamma, self.lam)

        # Advantage Pre-processing (Top-K Masking)
        # We only want to weight the "good" half of the samples.
        # This prevents the exponential weights from being dominated by outliers
        # and acts as a trust-region filter.
        with torch.no_grad():
            # 1. Select top-k advantages (top 50%)
            top_k_threshold = torch.quantile(adv, self.top_k_fraction)
            mask = adv >= top_k_threshold

            # 2. Extract selected advantages
            adv_selected = adv[mask]

            # 3. Centre for numerical stability ONLY
            adv_selected = adv_selected - adv_selected.mean()

            # 4. E-step weights
            eta = self.eta.exp()
            q_weights = torch.softmax(adv_selected / eta, dim=0)
            ess = 1.0 / torch.sum(q_weights**2)
            ess_frac = ess / q_weights.numel()

        # M-STEP & Value Update Loop
        # We iterate multiple times over the batch to fully regress the policy onto the target q.

        # Prepare datasets:
        # Policy uses ONLY Top-K data (masked)
        s_top = s[mask]
        a_top = a[mask]

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
            "entropy": entropy.item(),
            "rollout_return": float(np.sum(episode_returns))
            / max(1, len(episode_returns)),
            "policy_loss": total_policy_loss / self.n_policy_updates,  # M-step loss
            "value_loss": total_value_loss / self.n_value_updates,  # critic MSE
            "eta": eta.item(),  # current temperature
            "eta_loss": eta_loss.item(),  # dual objective value
            "alpha_mu": self.log_alpha_mu.exp().item(),
            "alpha_sigma": self.log_alpha_sigma.exp().item(),
            "kl_mu": total_kl_mu / self.n_policy_updates,
            "kl_sigma": total_kl_sigma / self.n_policy_updates,
            "kl": (total_kl_mu + total_kl_sigma) / self.n_policy_updates,
            "ess": ess.item(),
            "ess_frac": ess_frac.item(),
            "ess_ration": ess.item() / adv_selected.numel(),
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
        default=0.1,
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
        help="Random seed for reproducibility",
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
        "--top_k_fraction",
        type=float,
        default=1.0,
        help="Fraction of samples to keep based on advantage (top-k masking for trust-region filtering)",
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
        eps_alpha_mu=args.eps_alpha_mu,
        eps_alpha_sigma=args.eps_alpha_sigma,
        top_k_fraction=args.top_k_fraction,
    ).train(iters=args.iters)
    wandb.finish()
