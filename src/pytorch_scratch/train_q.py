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


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256), init_log_std=-1.0):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden) + [act_dim])
        self.log_std = nn.Parameter(init_log_std * torch.ones(act_dim))

    def forward(self, s):
        mu = self.net(s)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

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


class AdvantageCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim] + list(hidden) + [1])

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


def solve_eta(A, eps, n_iters=50):
    A = A - A.max()
    lo = torch.tensor(1e-4, device=A.device)
    hi = torch.tensor(100.0, device=A.device)

    for _ in range(n_iters):
        mid = (lo + hi) / 2
        scaled = A / mid
        w = torch.softmax(scaled, dim=0)
        logZ = torch.log(torch.mean(torch.exp(scaled)))
        g = eps + logZ - (w * A).sum() / mid
        lo, hi = (mid, hi) if g < 0 else (lo, mid)

    return ((lo + hi) / 2).clamp(1e-3, 100.0)


class VMPOTrainer:
    def __init__(
        self,
        env_name="Pendulum-v1",
        batch_size=2048,
        K=128,
        gamma=0.99,
        eps_eta=0.5,
        kl_beta=0.02,
    ):
        self.env = gym.make(env_name)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        self.actor = GaussianPolicy(obs_dim, act_dim).to(device)
        self.actor_old = deepcopy(self.actor)
        self.v = CriticV(obs_dim).to(device)
        self.v_target = deepcopy(self.v)
        self.adv = AdvantageCritic(obs_dim, act_dim).to(device)

        self.opt_pi = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_v = optim.Adam(self.v.parameters(), lr=3e-4)
        self.opt_adv = optim.Adam(self.adv.parameters(), lr=3e-4)

        self.batch_size = batch_size
        self.K = K
        self.gamma = gamma
        self.eps_eta = eps_eta
        self.kl_beta = kl_beta
        self.lambda_kl = 1.0

    def collect(self):
        obs, acts, rews, next_obs, dones = [], [], [], [], []
        ep_returns = []
        ep_return = 0.0

        s, _ = self.env.reset()

        for _ in range(self.batch_size):
            s_t = torch.tensor(s, dtype=torch.float32).to(device)
            with torch.no_grad():
                a = self.actor.sample(s_t).cpu().numpy()

            s2, r, d, tr, _ = self.env.step(a)

            obs.append(s)
            acts.append(a)
            rews.append(r)
            next_obs.append(s2)
            dones.append(float(d or tr))

            ep_return += r

            if d or tr:
                ep_returns.append(ep_return)
                ep_return = 0.0
                s, _ = self.env.reset()
            else:
                s = s2

        return (
            np.array(obs),
            np.array(acts),
            np.array(rews),
            np.array(next_obs),
            np.array(dones),
            ep_returns,
        )

    def train_once(self, it):
        obs, acts, rews, next_obs, dones, ep_returns = self.collect()
        s = torch.tensor(obs, dtype=torch.float32).to(device)
        a = torch.tensor(acts, dtype=torch.float32).to(device)
        r = torch.tensor(rews, dtype=torch.float32).to(device)
        s2 = torch.tensor(next_obs, dtype=torch.float32).to(device)
        d = torch.tensor(dones, dtype=torch.float32).to(device)

        # ---- V update
        with torch.no_grad():
            target_v = r + self.gamma * (1 - d) * self.v_target(s2)
        v_loss = ((self.v(s) - target_v) ** 2).mean()
        self.opt_v.zero_grad()
        v_loss.backward()
        self.opt_v.step()

        # ---- Advantage update
        with torch.no_grad():
            td_adv = target_v - self.v(s)
        adv_loss = ((self.adv(s, a) - td_adv) ** 2).mean()
        self.opt_adv.zero_grad()
        adv_loss.backward()
        self.opt_adv.step()

        # ---- E-step
        with torch.no_grad():
            mu, std = self.actor_old(s)
            eps = torch.randn(len(s), self.K, a.shape[1], device=device)
            a_k = torch.tanh(mu[:, None] + eps * std[:, None])

            s_rep = s[:, None].expand(-1, self.K, -1).reshape(-1, s.shape[1])
            a_rep = a_k.reshape(-1, a.shape[1])

            A = self.adv(s_rep, a_rep).view(len(s), self.K)
            A = (A - A.mean(dim=1, keepdim=True)) / (A.std(dim=1, keepdim=True) + 1e-6)

        etas = torch.stack([solve_eta(A[i], self.eps_eta) for i in range(len(A))])
        w = torch.softmax(A / etas[:, None], dim=1)
        flat_w = w.reshape(-1).detach()

        # ---- M-step
        kl_unweighted = 0.0
        n_policy_steps = 10
        for i in range(n_policy_steps):
            logp = self.actor.log_prob(s_rep, a_rep)
            loss_nll = -(flat_w * logp).mean()

            old = self.actor_old.log_prob(s_rep, a_rep)
            kl = (flat_w * (old - logp)).sum()
            if i == n_policy_steps - 1:
                with torch.no_grad():
                    kl_unweighted = (old - logp).mean().item()

            loss = loss_nll + self.lambda_kl * kl
            self.opt_pi.zero_grad()
            loss.backward()
            self.opt_pi.step()
        self.lambda_kl *= math.exp(0.2 * (kl_unweighted - self.kl_beta))
        self.lambda_kl = float(np.clip(self.lambda_kl, 1e-4, 100.0))

        self.actor_old.load_state_dict(self.actor.state_dict())

        return {
            "v_loss": v_loss.item(),
            "adv_loss": adv_loss.item(),
            "eta_mean": etas.mean().item(),
            "posterior_entropy": (-w * torch.log(w + 1e-8)).sum(dim=1).mean().item(),
            "ep_return_mean": np.mean(ep_returns) if ep_returns else 0.0,
            "ep_return_max": np.max(ep_returns) if ep_returns else 0.0,
        }

    def train(self, iters=200):
        for it in range(iters):
            info = self.train_once(it)
            wandb.log(info)
            print(it, info)


if __name__ == "__main__":
    wandb.init(project="vmpo-advantage-final")
    VMPOTrainer().train()
    wandb.finish()
