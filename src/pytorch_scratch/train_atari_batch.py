# V-MPO (On-Policy Maximum a Posteriori Policy Optimisation) for Atari

import numpy as np
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_atari_env(game, seed=None):
    """Create an Atari environment with standard DeepMind-style preprocessing.

    Wraps the raw ALE environment with:
      - AtariPreprocessing: frame-skip of 4 (action repeated 4 frames),
        grayscale conversion, observation scaling to [0,1], and episode
        termination on life loss (helps early training signal).
      - FrameStackObservation: stacks the last 4 frames along the channel
        dimension so the agent can perceive motion / velocity.

    Sticky actions (repeat_action_probability=0.25) add stochasticity to the
    environment to reduce over-fitting to deterministic dynamics.
    """
    env = gym.make(
        f"ALE/{game}-v5",
        frameskip=1,  # raw env emits every frame
        repeat_action_probability=0.25,  # 25% chance previous action is repeated (sticky actions)
        full_action_space=False,  # use minimal action set for this game
    )

    env = AtariPreprocessing(
        env,
        frame_skip=4,  # agent sees every 4th frame
        grayscale_obs=True,  # convert RGB → single-channel grayscale
        scale_obs=True,  # scale pixel values to [0, 1]
        terminal_on_life_loss=True,  # treat life loss as episode end
    )

    # Stack 4 consecutive (already frame-skipped) observations → (4, 84, 84)
    env = FrameStackObservation(env, stack_size=4)

    if seed is not None:
        env.reset(seed=seed)

    return env


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

    # Walk backwards through the rollout to accumulate the
    # exponentially-weighted sum of TD errors.
    for t in reversed(range(T)):
        # 1-step TD error: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        # (1 - done) masks the bootstrap when an episode ended at step t.
        delta = rewards[t] + gamma * (1 - dones[t]) * values[t + 1] - values[t]
        # Recursive GAE accumulation: A_t = δ_t + γλ · A_{t+1}
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    # Value-function regression targets: R_t = A_t + V(s_t)
    returns = advantages + values[:-1]
    return advantages, returns


# Policy Network  (π_θ)
# Outputs a categorical distribution over the discrete action space.
# Architecture follows the classic "Nature DQN" CNN backbone:
#   Conv(4→32, 8×8, stride 4) → Conv(32→64, 4×4, stride 2) →
#   Conv(64→64, 3×3, stride 1) → FC(3136→512) → FC(512→n_actions)
# The final layer produces raw logits (un-normalised log-probabilities).
class CategoricalPolicy(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),  # (4, 84, 84) → (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # (32, 20, 20) → (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),  # (64, 9, 9)  → (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # → 64*7*7 = 3136
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),  # logits over actions
        )

    def forward(self, x):
        """Return raw logits for each action."""
        return self.net(x)

    def sample(self, x):
        """Sample an action from the categorical distribution."""
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

    def log_prob(self, x, a):
        """Compute log π_θ(a | s) for given state-action pairs."""
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(a)


# Value Network  V_φ(s)
# Same CNN backbone as the policy but with a single scalar output that
# estimates the state-value function.  A separate network (not shared with
# the policy) is used to avoid interference between value and policy
# gradients.
class AtariValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1),  # scalar value estimate
        )

    def forward(self, x):
        """Return V(s) as a scalar per batch element."""
        return self.net(x).squeeze(-1)


# VMPO Trainer
# Orchestrates the full VMPO training loop:
#   - Maintains two copies of the policy: π_θ (updated) and π_old (behaviour).
#   - Maintains a learned temperature η (stored in log-space for positivity).
#   - Each iteration: collect → GAE → E-step → M-step → dual η update →
#     value update → sync π_old ← π_θ.
class AtariTrainer:
    def __init__(
        self,
        game="Pong",
        rollout_steps=4096,
        gamma=0.99,
        lam=0.95,
        lr=2.5e-4,
        n_temperature_epsilon=0.1,
        eta_initial=0.0,
    ):
        self.env = make_atari_env(game)

        # π_θ  – the policy we optimise (M-step target)
        self.policy = CategoricalPolicy(self.env.action_space.n).to(device)
        # π_old – frozen copy used as the behaviour policy for data collection
        #         and as the reference distribution for KL measurement.
        self.policy_old = copy.deepcopy(self.policy).eval()
        # V_φ  – state-value function (fitted during the value-update step)
        self.value = AtariValue().to(device)

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

        self.eps_alpha = 0.01  # KL bound ε_α (Atari-scale safe default)

        # Running episode statistics (accumulated across rollout boundaries)
        self.ep_return = 0.0
        self.ep_length = 0
        self.total_env_steps = 0

        self.target_update_interval = 4  # T_target (Atari-safe default)
        self._target_update_counter = 0

    # Data Collection  (rollout with π_old)

    def collect(self):
        """Collect a fixed-length rollout using the behaviour policy π_old.

        Returns tensors of observations, actions, rewards, done flags,
        value estimates (T+1 entries – includes bootstrap), and a list of
        completed episode returns for logging.
        """
        obs, acts, rews, dones, vals = [], [], [], [], []

        s, _ = self.env.reset()

        episode_returns = []  # returns of episodes that completed during this rollout

        for _ in range(self.rollout_steps):
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)

            # Sample action from the *behaviour* policy π_old (no grad needed)
            # and record the value estimate V(s_t) for GAE.
            with torch.no_grad():
                a = self.policy_old.sample(s_t).item()
                v = self.value(s_t).item()

            s2, r, terminated, truncated, _ = self.env.step(a)
            self.total_env_steps += 1
            done = terminated or truncated

            # Reward clipping to [-1, 1] (standard Atari normalisation)
            r = np.clip(r, -1.0, 1.0)

            # Accumulate episode-level statistics
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
            torch.from_numpy(np.asarray(acts, dtype=np.int64)).to(device),
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
        q_top = q_weights[mask]

        # Value function uses ALL data (unmasked)
        s_full = s
        ret_full = ret

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0

        for epoch in range(n_epochs):
            # --- Policy Update Loop (on Top-K data) ---

            idx = torch.randperm(len(s_top))
            mb_s = s_top[idx]
            mb_a = a_top[idx]
            mb_w = q_top[idx]

            # M-STEP: Weighted Maximum Likelihood
            # New policy
            logits = self.policy(mb_s)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(mb_a)

            # Old policy (detached)
            with torch.no_grad():
                old_logits = self.policy_old(mb_s)
                old_dist = torch.distributions.Categorical(logits=old_logits)

            # L_pi = - sum( w_i * log pi(a|s) )
            # Note: mb_w sum is not 1 here due to mini-batching, but gradient scales linearly.
            # Since weights sum to 1 over the full set, we sum here (not mean).
            policy_loss = -(mb_w * logp).sum()

            # KL(π_old || π_new)
            kl = torch.distributions.kl_divergence(old_dist, dist).mean()
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

        # Sync Behaviour Policy  π_old ← π_θ
        # The updated policy becomes the behaviour policy for the next
        # rollout.  This is the "hard" target-network update.
        self._target_update_counter += 1

        if self._target_update_counter % self.target_update_interval == 0:
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
            # Check entropy on a subset to save compute
            new_logits = self.policy(s_full[:512])
            new_dist = torch.distributions.Categorical(logits=new_logits)
            entropy = new_dist.entropy().mean()

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
    parser = argparse.ArgumentParser(description="Train Atari with VMPO")
    parser.add_argument(
        "--game",
        type=str,
        default="Pong",
        help="ALE game name (e.g. Pong, Breakout, SpaceInvaders)",
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
        default=2.5e-4,
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

    # Initialise Weights & Biases experiment tracking
    wandb.init(project="atari-pong-baseline", config=vars(args))
    AtariTrainer(
        game=args.game,
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        lam=args.lam,
        lr=args.lr,
        n_temperature_epsilon=args.n_temperature_epsilon,
        eta_initial=args.eta_initial,
    ).train(iters=args.iters)
    wandb.finish()
