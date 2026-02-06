# V-MPO (On-Policy Maximum a Posteriori Policy Optimisation) for Atari
#
# V-MPO is a two-step EM-style policy-optimisation algorithm:
#
#   E-step  – Compute non-parametric, advantage-weighted sample distribution.
#             A temperature parameter η (eta) controls how "selective" the
#             weighting is.  η is itself optimised via a dual objective that
#             enforces a soft KL constraint (ε_η) on the sample weights.
#
#   M-step  – Fit the parametric policy π_θ to the weighted samples produced
#             by the E-step via supervised (weighted) maximum-likelihood.
#
# The overall loop is:
#   Collect rollout data with the current behaviour policy π_old.
#   Estimate advantages with GAE.
#   E-step: form sample weights from advantages / η  (softmax).
#   M-step: minimise  -Σ w_i · log π_θ(a_i | s_i).
#   Dual update: optimise η to keep the weight distribution close to
#      uniform (bounded information loss).
#   Value update: regress V(s) towards the GAE returns.
#   Copy π_θ → π_old for the next iteration.


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
        eta_initial=1.0,
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

        # Running episode statistics (accumulated across rollout boundaries)
        self.ep_return = 0.0
        self.ep_length = 0
        self.total_env_steps = 0

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
            done = terminated or truncated

            # Reward clipping to [-1, 1] (standard Atari normalisation)
            r = np.clip(r, -1.0, 1.0)

            # Accumulate episode-level statistics
            self.ep_return += r
            self.ep_length += 1

            if done:
                episode_returns.append(self.ep_return)
                self.ep_return = 0.0
                self.ep_length = 0
                s, _ = self.env.reset()
            else:
                s = s2

            obs.append(s)
            acts.append(a)
            rews.append(r)
            dones.append(float(done))
            vals.append(v)
            self.total_env_steps += 1
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

        with torch.no_grad():
            # Calculate top 50% threshold
            top_k_threshold = torch.quantile(adv, 0.5)
            mask = adv >= top_k_threshold
            
            # Select only top-k samples for probability calculation
            adv_selected = adv[mask]
            
            # Centre advantages for numerical stability of softmax
            # (Note: V-MPO formulation is invariant to shifting A, but float precision isn't)
            adv_selected = adv_selected - adv_selected.mean()
            
            # E-STEP: Compute weights on the subset
            eta = self.eta.exp()
            # w_i = exp(A_i / eta) / Z
            weights = torch.softmax(adv_selected / eta, dim=0)

        # Snapshot the behaviour policy's logits (needed for KL after
        # the M-step and for diagnostics).
        with torch.no_grad():
            old_logits = self.policy_old(s)

        # M-STEP  –  weighted maximum-likelihood policy update
        # Minimise  L_π = - Σ_i  w_i · log π_θ(a_i | s_i)
        # where w_i are the E-step weights (detached – treated as
        # constants w.r.t. θ).  This is equivalent to fitting π_θ to
        # the advantage-weighted sample distribution q from the E-step.

        new_logits_pre = self.policy(s)
        new_dist_pre = torch.distributions.Categorical(logits=new_logits_pre)
        logp = new_dist_pre.log_prob(a)

        policy_loss = -(weights.detach() * logp).mean()

        self.opt_pi.zero_grad()
        policy_loss.backward()
        self.opt_pi.step()

        # KL Divergence Measurement  (diagnostic, post M-step)

        # Measure KL( π_old || π_θ ) *after* the policy gradient step
        # to monitor how much the policy changed this iteration.
        with torch.no_grad():
            new_logits = self.policy(s)
            max_logit_diff = (new_logits - old_logits).abs().max().item()

        old_dist = torch.distributions.Categorical(logits=old_logits)
        new_dist = torch.distributions.Categorical(logits=new_logits)
        kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

        # Sync Behaviour Policy  π_old ← π_θ

        # The updated policy becomes the behaviour policy for the next
        # rollout.  This is the "hard" target-network update.
        self.policy_old.load_state_dict(self.policy.state_dict())

        # DUAL η UPDATE  –  E-step temperature optimisation

        # η is optimised by minimising the dual function that enforces
        # the KL constraint  KL(q || uniform) ≤ ε_η  on the E-step
        # weight distribution.
        #
        # The dual objective (derived via Lagrangian relaxation) is:
        #   g(η) = η · ε_η  +  η · log( (1/T) Σ_i exp(A_i / η) )
        #        = η · [ ε_η  +  logsumexp(A / η) - log T ]
        #
        # Minimising g(η) w.r.t. η tightens or loosens the temperature
        # to satisfy the constraint.  Advantages are detached so the
        # gradient flows only through η.

        T = adv.numel()

        eta_loss = eta * (
            self.n_temperature_epsilon
            + torch.logsumexp(adv.detach() / eta, dim=0)
            - torch.log(torch.tensor(float(T), device=adv.device))
        )

        self.opt_eta.zero_grad()
        eta_loss.backward()
        self.opt_eta.step()

        # VALUE UPDATE  –  policy evaluation / critic fitting
        # Standard MSE regression of V_φ(s) towards the GAE returns R_t.
        # This is independent of the E/M steps and simply improves the
        # baseline for the next iteration's advantage estimates.

        value_loss = ((self.value(s) - ret) ** 2).mean()

        self.opt_v.zero_grad()
        value_loss.backward()
        self.opt_v.step()

        # Diagnostics / Logging

        with torch.no_grad():
            entropy = new_dist.entropy().mean()  # policy entropy (exploration health)

        metrics = {
            "entropy": entropy.item(),
            "rollout_return": r.sum().item(),  # total reward in this rollout
            "policy_loss": policy_loss.item(),  # M-step loss
            "value_loss": value_loss.item(),  # critic MSE
            "eta": eta.item(),  # current temperature
            "eta_loss": eta_loss.item(),  # dual objective value
            "kl": float(kl.cpu()),  # KL(π_old || π_θ)
            "kl_sci": float(f"{kl.item():.6e}"),  # KL in scientific notation
            "max_logit_diff": max_logit_diff,  # largest per-action logit change
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
        default=1024,
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
        default=1.0,
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
