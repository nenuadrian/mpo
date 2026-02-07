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


class PopArt:
    """
    Single-output PopArt for single-task value normalisation.
    - Tracks raw moments M1, M2 via EMA with factor beta.
    - Provides `update()` to change stats and reparam the final linear layer,
      adjusting optimizer state if provided.
    """

    def __init__(self, device="cpu", beta=1e-3, eps_var=1e-6, sigma_min=1e-2):
        self.device = torch.device(device)
        self.beta = float(beta)
        self.eps_var = float(eps_var)
        self.sigma_min = float(sigma_min)

        # raw moments (EMA)
        self.M1 = torch.tensor(0.0, device=self.device)
        self.M2 = torch.tensor(self.eps_var, device=self.device)

        # derived stats
        self.mu = torch.tensor(0.0, device=self.device)
        self.sigma = torch.tensor(1.0, device=self.device)

    @torch.no_grad()
    def stats(self):
        return float(self.mu), float(self.sigma)

    @torch.no_grad()
    def update(
        self, targets, value_head: nn.Linear, optimizer: torch.optim.Optimizer = None
    ):
        """
        Update running moments from `targets` (1D tensor). Rescale `value_head` in-place to preserve unnormalised outputs.
        If optimizer is provided (Adam), try to rescale param-state entries for value_head.weight and value_head.bias.

        Usage: call this AFTER you computed the raw returns (G) but BEFORE computing normalised targets for the value loss.
        """
        if targets.numel() == 0:
            return

        targets = targets.detach().to(self.device).view(-1)

        batch_mean = targets.mean()
        batch_msq = (targets * targets).mean()

        # store old stats
        mu_old = self.M1.clone()
        sigma_old = self.sigma.clone()

        # update raw moments
        self.M1 = (1.0 - self.beta) * self.M1 + self.beta * batch_mean
        self.M2 = (1.0 - self.beta) * self.M2 + self.beta * batch_msq

        # derive new stats
        mu_new = self.M1
        var_new = (self.M2 - self.M1 * self.M1).clamp(min=self.eps_var)
        sigma_new = var_new.sqrt().clamp(min=self.sigma_min)

        # compute transform
        c = (sigma_old / sigma_new).to(self.device)
        d = ((mu_old - mu_new) / sigma_new).to(self.device)

        # apply reparameterisation to final linear layer
        with torch.no_grad():
            # value_head.weight shape (1, hidden) and bias shape (1,)
            value_head.weight.mul_(c)
            value_head.bias.mul_(c)
            value_head.bias.add_(d)

        # adjust optimizer state for that param if provided (Adam-like)
        if optimizer is not None:
            params = {
                id(p): p for group in optimizer.param_groups for p in group["params"]
            }
            # check both weight and bias
            for p in (value_head.weight, value_head.bias):
                st = optimizer.state.get(p)
                if st is None:
                    continue
                if "exp_avg" in st:
                    st["exp_avg"].mul_(c)
                if "exp_avg_sq" in st:
                    st["exp_avg_sq"].mul_(c * c)

        # save derived stats
        self.mu = mu_new
        self.sigma = sigma_new

    def normalize(self, targets):
        """Return normalised targets (G - mu) / sigma as a tensor on self.device."""
        return (targets.to(self.device) - self.mu) / self.sigma

    def unnormalise_from_norm(self, v_norm):
        """Convert normalised network outputs to unnormalised scalars: v = sigma * v_norm + mu"""
        return self.sigma * v_norm + self.mu


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

    # GAE advantage A^\lambda_t and corresponding \lambda-return target G^\lambda_t
    lambda_returns = advantages + values
    return advantages, lambda_returns


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
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


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
        top_k_fraction: float,
        n_value_updates: int,
        n_policy_updates: int,
        eta_lr: float,
        t_target: int,
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
        self.top_k_fraction = top_k_fraction

        self.n_value_updates = n_value_updates
        self.n_policy_updates = n_policy_updates

        self.domain = domain
        self.task = task

        self.popart = PopArt(device=device, beta=3e-4, eps_var=1e-6, sigma_min=1e-4)

        self.learn_iter = 0
        self.T_target = t_target

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
                # network outputs normalised value; convert to unnormalised scalar for GAE
                v_norm = self.value(s_t)
                v = (v_norm * self.popart.sigma + self.popart.mu).item()

            s2, r, terminated, truncated, _ = self.env.step(a)
            s2 = flatten_obs(s2)
            self.total_env_steps += 1

            # Gymnasium gives both for a reason:
            # - terminated => true terminal => do NOT bootstrap
            # - truncated  => time limit    => DO bootstrap, but still reset
            bootstrap_mask = 0.0 if terminated else 1.0

            with torch.no_grad():
                s2_t = torch.tensor(s2, dtype=torch.float32).unsqueeze(0).to(device)
                v_next_norm = self.value(s2_t)
                v_next = (v_next_norm * self.popart.sigma + self.popart.mu).item()

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
        self.learn_iter += 1
        # Data Collection  –  rollout with behaviour policy π_old
        s, a, r, bootstrap_masks, v, v_next, episode_returns = self.collect()

        # Advantage Estimation (GAE-λ)
        adv, lambda_ret = compute_gae(
            r, v, v_next, bootstrap_masks, self.gamma, self.lam
        )

        # Advantage Pre-processing (Top-K Masking)
        # We only want to weight the "good" half of the samples.
        # This prevents the exponential weights from being dominated by outliers
        # and acts as a trust-region filter.
        with torch.no_grad():
            # 1. Select top-k advantages (top 50%)
            if self.top_k_fraction < 1.0:
                top_k_threshold = torch.quantile(adv, self.top_k_fraction)
                mask = adv >= top_k_threshold
            else:
                mask = torch.ones_like(adv, dtype=torch.bool)

            # 2. Extract selected advantages
            adv_selected = adv[mask]
            s_top = s[mask]
            a_top = a[mask]

            # 3. Centre for numerical stability ONLY
            if self.total_env_steps < 1_000_000:
                adv_selected = adv_selected - adv_selected.mean()

            # 4. E-step weights

            eta = self.eta.exp()

            q_weights = torch.softmax((adv_selected / eta), dim=0)
            ess = 1.0 / torch.sum(q_weights**2)
            ess_frac = ess / q_weights.numel()

        # M-STEP & Value Update Loop
        # We iterate multiple times over the batch to fully regress the policy onto the target q.

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

        # --- PopArt: update stats & reparam final layer BEFORE value regression ---
        # 'lambda_ret' contains unnormalised \lambda-returns. Update PopArt using these
        # so mu/sigma are current and the value head is reparameterised.
        value_head = self.value.head
        self.popart.update(lambda_ret.detach(), value_head, optimizer=self.opt_v)

        # now compute normalised targets for the value regression
        ret_norm = self.popart.normalize(lambda_ret)

        # --- Value Update Loop (on ALL data) using normalised targets ---
        total_value_loss = 0.0
        for _ in range(self.n_value_updates):
            values_norm = self.value(s)  # network outputs normalised values
            value_loss = ((values_norm - ret_norm) ** 2).mean()

            self.opt_v.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.opt_v.step()

            total_value_loss += value_loss.item()

        if self.learn_iter % self.T_target == 0:
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
            "train/ess_ratio": ess.item() / adv_selected.numel(),
            "popart/mu": self.popart.mu.item(),
            "popart/sigma": self.popart.sigma.item(),
            "debug/value_mean": v.mean().item(),
            "debug/ret_mean": lambda_ret.mean().item(),
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
            "popart_mu": self.popart.mu.detach().cpu(),
            "popart_sigma": self.popart.sigma.detach().cpu(),
            "popart_M1": self.popart.M1.detach().cpu(),
            "popart_M2": self.popart.M2.detach().cpu(),
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
        default=0.1,
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
        default=0.5,
        help="Fraction of samples to keep based on advantage (top-k masking for trust-region filtering)",
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
    parser.add_argument(
        "--t_target",
        type=int,
        default=7,
    )
    args = parser.parse_args()

    set_seed(args.seed)

    wandb.init(  # type: ignore
        project="dm-control-vmpo",
        config=vars(args),
        group=f"{args.domain}-{args.task}",
        name=f"{args.domain}-{args.task}-vmpo-eta{args.eta_initial}-topk{args.top_k_fraction}-eps{args.n_temperature_epsilon}-seed{args.seed}",
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
        top_k_fraction=args.top_k_fraction,
        n_value_updates=args.n_value_updates,
        n_policy_updates=args.n_policy_updates,
        eta_lr=args.eta_lr,
        t_target=args.t_target,
    ).train(iters=args.iters)
    wandb.finish()  # type: ignore
