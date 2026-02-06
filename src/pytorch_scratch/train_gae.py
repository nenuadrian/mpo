import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import wandb
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


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


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden) + [act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, s):
        mu = self.net(s)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, torch.exp(log_std)

    def sample(self, s):
        mu, std = self.forward(s)
        eps = torch.randn_like(mu)
        return torch.tanh(mu + eps * std)

    def log_prob(self, s, a):
        mu, std = self.forward(s)
        pre_tanh = torch.atanh(a.clamp(-0.999, 0.999))
        eps = (pre_tanh - mu) / std
        logp = -0.5 * (eps**2 + 2 * torch.log(std) + math.log(2 * math.pi))
        logp = logp.sum(-1)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return logp


class CriticV(nn.Module):
    def __init__(self, obs_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden) + [1])

    def forward(self, s):
        return self.net(s).squeeze(-1)


def solve_eta(A, eps, n_iters=50):
    A = A - A.max()
    lo, hi = 1e-4, 100.0

    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        scaled = A / mid
        w = torch.softmax(scaled, dim=0)
        logZ = torch.log(torch.mean(torch.exp(scaled)))
        g = eps + logZ - (w * A).sum() / mid
        if g < 0:
            hi = mid
        else:
            lo = mid

    return torch.tensor(0.5 * (lo + hi), device=A.device)


class VMPOTrainer:
    def __init__(
        self,
        env_name="Pendulum-v1",
        batch_size=2048,
        K=64,
        gamma=0.99,
        lam=0.95,
        eps_eta=0.5,
        kl_beta=0.02,
    ):
        self.env = gym.make(env_name)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        self.actor = GaussianPolicy(obs_dim, act_dim).to(device)
        self.actor_old = deepcopy(self.actor)
        self.v = CriticV(obs_dim).to(device)

        self.opt_pi = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_v = optim.Adam(self.v.parameters(), lr=3e-4)

        self.batch_size = batch_size
        self.K = K
        self.gamma = gamma
        self.lam = lam
        self.eps_eta = eps_eta
        self.kl_beta = kl_beta
        self.lambda_kl = 1.0

    def collect(self):
        obs, acts, rews, dones = [], [], [], []
        s, _ = self.env.reset()

        for _ in range(self.batch_size):
            s_t = torch.tensor(s, dtype=torch.float32).to(device)
            with torch.no_grad():
                a = self.actor_old.sample(s_t).cpu().numpy()

            s2, r, d, tr, _ = self.env.step(a)

            obs.append(s)
            acts.append(a)
            rews.append(r)
            dones.append(float(d or tr))

            s = s2 if not (d or tr) else self.env.reset()[0]

        return (
            torch.tensor(obs, dtype=torch.float32).to(device),
            torch.tensor(acts, dtype=torch.float32).to(device),
            torch.tensor(rews, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device),
        )

    def train_once(self):
        s, a, r, d = self.collect()

        with torch.no_grad():
            v = self.v(s)
            v_next = torch.cat([v[1:], v[-1:]])
            values_ext = torch.cat([v, v_next[-1:]])

        _, ret = compute_gae(r, values_ext, d, self.gamma, self.lam)

        # ---- value update
        v_loss = ((self.v(s) - ret) ** 2).mean()
        self.opt_v.zero_grad()
        v_loss.backward()
        self.opt_v.step()

        # ---- E-step
        with torch.no_grad():
            mu, std = self.actor_old(s)
            eps = torch.randn(len(s), self.K, a.shape[1], device=device)
            a_k = torch.tanh(mu[:, None] + eps * std[:, None])

            s_rep = s[:, None].expand(-1, self.K, -1).reshape(-1, s.shape[1])
            a_rep = a_k.reshape(-1, a.shape[1])

            logp_k = self.actor_old.log_prob(s_rep, a_rep).view(len(s), self.K)

            A = logp_k - logp_k.mean(dim=1, keepdim=True)
            A = A + 0.1 * ret.unsqueeze(1)

            k = max(1, self.K // 2)  # top 50%
            topk_vals, _ = torch.topk(A, k, dim=1)
            threshold = topk_vals[:, -1].unsqueeze(1)

            A = torch.where(A >= threshold, A, -1e9)

        etas = torch.stack([solve_eta(A[i], self.eps_eta) for i in range(len(A))])
        w = torch.softmax(A / etas[:, None], dim=1).detach()
        flat_w = w.reshape(-1)

        # ---- M-step
        n_policy_steps = 5
        for m in range(n_policy_steps):
            logp = self.actor.log_prob(s_rep, a_rep)
            old_logp = self.actor_old.log_prob(s_rep, a_rep)

            loss_pi = -(flat_w * logp).mean()
            kl = (old_logp - logp).mean()

            loss = loss_pi + self.lambda_kl * kl

            self.opt_pi.zero_grad()
            loss.backward()
            self.opt_pi.step()

        self.lambda_kl *= math.exp(0.2 * (kl.item() - self.kl_beta))
        self.lambda_kl = float(np.clip(self.lambda_kl, 1e-4, 100.0))

        self.actor_old.load_state_dict(self.actor.state_dict())

        return {
            "return": ret.mean().item(),
            "v_loss": v_loss.item(),
            "policy_loss": loss_pi.item(),
            "kl": kl.item(),
            "eta_mean": etas.mean().item(),
        }

    def train(self, iters=200):
        for it in range(iters):
            info = self.train_once()
            wandb.log(info)
            print(it, info)


if __name__ == "__main__":
    wandb.init(project="vmpo-gae-final")
    VMPOTrainer().train()
    wandb.finish()
