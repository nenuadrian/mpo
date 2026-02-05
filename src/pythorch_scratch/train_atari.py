import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_atari_env(game, seed=None):
    env = gym.make(
        f"ALE/{game}-v5",
        frameskip=1,
        repeat_action_probability=0.25,
        full_action_space=False,
    )

    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=True,
        terminal_on_life_loss=True,
    )

    env = FrameStackObservation(env, stack_size=4)

    if seed is not None:
        env.reset(seed=seed)

    return env


# ---------------------------
# GAE
# ---------------------------
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T, device=values.device)
    gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * (1 - dones[t]) * values[t + 1] - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


class CategoricalPolicy(nn.Module):
    def __init__(self, n_actions):
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
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x)

    def sample(self, x):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

    def log_prob(self, x, a):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(a)


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
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AtariTrainer:
    def __init__(
        self,
        game="Pong",
        rollout_steps=1024,
        gamma=0.99,
        lam=0.95,
        lr=2.5e-4,
    ):
        self.env = make_atari_env(game)
        self.policy = CategoricalPolicy(self.env.action_space.n).to(device)
        self.policy_old = copy.deepcopy(self.policy).eval()
        self.value = AtariValue().to(device)

        self.opt_pi = optim.Adam(self.policy.parameters(), lr=lr)
        self.opt_v = optim.Adam(self.value.parameters(), lr=lr)

        self.eta = nn.Parameter(torch.tensor(1.0, device=device))
        self.opt_eta = optim.Adam([self.eta], lr=1e-3)

        self.rollout_steps = rollout_steps
        self.gamma = gamma
        self.lam = lam

        self.ep_return = 0.0
        self.ep_length = 0

    def collect(self):
        obs, acts, rews, dones, vals = [], [], [], [], []
        # share of obs

        s, _ = self.env.reset()

        episode_returns = []

        for _ in range(self.rollout_steps):
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                a = self.policy_old.sample(s_t).item()
                v = self.value(s_t).item()

            s2, r, terminated, truncated, _ = self.env.step(a)
            done = terminated or truncated

            r = np.clip(r, -1.0, 1.0)

            # episode tracking
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

        with torch.no_grad():
            v_last = self.value(
                torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
            ).item()

        vals.append(v_last)

        return (
            torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(acts, dtype=np.int64)).to(device),
            torch.from_numpy(np.asarray(rews, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(dones, dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(vals, dtype=np.float32)).to(device),
            episode_returns,
        )

    def train_once(self):
        # collect rollout with behaviour policy (policy_old)
        s, a, r, d, v, episode_returns = self.collect()

        # advantages and returns (for policy_old)
        adv, ret = compute_gae(r, v, d, self.gamma, self.lam)

        # centre advantages (VMPO requires relative scale)
        adv = adv - adv.mean()

        # VMPO E-step weights
        eta = self.eta.exp()
        weights = torch.softmax(adv / eta, dim=0)

        # freeze reference policy logits
        with torch.no_grad():
            old_logits = self.policy_old(s)

        # M-step: policy update (weighted MLE)
        new_logits_pre = self.policy(s)
        new_dist_pre = torch.distributions.Categorical(logits=new_logits_pre)
        logp = new_dist_pre.log_prob(a)

        policy_loss = -(weights.detach() * logp).mean()

        self.opt_pi.zero_grad()
        policy_loss.backward()
        self.opt_pi.step()

        # measure KL AFTER policy update
        with torch.no_grad():
            new_logits = self.policy(s)

        old_dist = torch.distributions.Categorical(logits=old_logits)
        new_dist = torch.distributions.Categorical(logits=new_logits)
        kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

        # now update reference policy (becomes next behaviour policy)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # η dual optimisation (E-step temperature)
        epsilon = 0.1
        T = adv.numel()

        eta_loss = eta * (
            epsilon
            + torch.logsumexp(adv.detach() / eta, dim=0)
            - torch.log(torch.tensor(float(T), device=adv.device))
        )

        self.opt_eta.zero_grad()
        eta_loss.backward()
        self.opt_eta.step()

        # value update (policy evaluation)
        value_loss = ((self.value(s) - ret) ** 2).mean()

        self.opt_v.zero_grad()
        value_loss.backward()
        self.opt_v.step()

        # diagnostics
        with torch.no_grad():
            entropy = new_dist.entropy().mean()

        metrics = {
            "entropy": entropy.item(),
            "rollout_return": r.sum().item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "eta": eta.item(),
            "eta_loss": eta_loss.item(),
            "kl": float(kl.cpu()),
            "kl_sci": float(f"{kl.item():.6e}"),
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
        for it in range(iters):
            info = self.train_once()
            wandb.log(info)
            if it % 10 == 0:
                print(it, info)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    wandb.init(project="atari-pong-baseline")
    AtariTrainer(game="Pong").train()
    wandb.finish()
