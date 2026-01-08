import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# Try to reuse dm_control helpers from the MPO script; fall back if imports fail.
try:
    from acme_pytorch_sync.train import make_env, flatten_observation, get_env_specs  # type: ignore
except Exception:
    from dm_control import suite  # type: ignore

    def make_env(env_name: str, render_mode: str | None = None):
        domain, task = env_name.split("::")
        env = suite.load(domain, task)

        class _Wrapper:
            def __init__(self, e):
                self._env = e

            def reset(self):
                ts = self._env.reset()
                return ts.observation, {}

            def step(self, action):
                ts = self._env.step(action)
                obs = ts.observation
                reward = float(ts.reward or 0.0)
                terminated = bool(ts.last())
                truncated = False
                return obs, reward, terminated, truncated, {}

            def close(self):
                self._env.close()

        return _Wrapper(env)

    def flatten_observation(obs):
        if isinstance(obs, dict):
            parts = [np.asarray(v).ravel() for v in obs.values()]
            return np.concatenate(parts).astype(np.float32)
        return np.asarray(obs).ravel().astype(np.float32)

    def get_env_specs(env_name: str) -> Tuple[int, int, np.ndarray, np.ndarray]:
        domain, task = env_name.split("::")
        probe_env = suite.load(domain, task)
        obs_spec = probe_env.observation_spec()
        if isinstance(obs_spec, dict):
            obs_dim = sum(int(np.prod(sp.shape)) for sp in obs_spec.values())
        else:
            obs_dim = int(np.prod(obs_spec.shape))
        action_spec = probe_env.action_spec()
        action_dim = int(np.prod(action_spec.shape))
        action_low = np.array(action_spec.minimum, dtype=np.float32)
        action_high = np.array(action_spec.maximum, dtype=np.float32)
        probe_env.close()
        return obs_dim, action_dim, action_low, action_high


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    # True where attention should be blocked; shape [T, T]
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


class TrajectoryTransformerPolicy(nn.Module):
    """
    Causal Transformer over tokens defined by (obs_t, prev_action_t, reward_t, timestep_t).
    Outputs mean action for each t (supervised BC).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_timestep: int = 10000,
    ):
        super().__init__()
        self.obs_emb = nn.Linear(obs_dim, d_model)
        self.act_emb = nn.Linear(action_dim, d_model)
        self.rew_emb = nn.Linear(1, d_model)
        self.tok_ln = nn.LayerNorm(d_model)

        self.timestep_emb = nn.Embedding(max_timestep, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_dim),
        )

        # Small init for head helps stability early on.
        with torch.no_grad():
            nn.init.uniform_(self.head[-1].weight, a=-1e-4, b=1e-4)
            nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        obs: torch.Tensor,  # [B,T,obs_dim]
        prev_action: torch.Tensor,  # [B,T,act_dim]
        reward: torch.Tensor,  # [B,T,1]
        timestep: torch.Tensor,  # [B,T] (int64)
    ) -> torch.Tensor:
        x = self.obs_emb(obs) + self.act_emb(prev_action) + self.rew_emb(reward)
        x = x + self.timestep_emb(timestep.clamp_min(0))
        x = self.tok_ln(x)

        T = x.shape[1]
        attn_mask = causal_mask(T, x.device)  # [T,T] bool
        h = self.tr(x, mask=attn_mask)
        return self.head(h)  # [B,T,act_dim]


@dataclass
class Trajectory:
    obs: np.ndarray  # [L, obs_dim]
    act: np.ndarray  # [L, act_dim]
    rew: np.ndarray  # [L, 1]
    done: np.ndarray  # [L, 1]
    t: np.ndarray  # [L,]


class EpisodeBuffer:
    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = int(max_episodes)
        self.episodes: List[Trajectory] = []

    def add(self, traj: Trajectory) -> None:
        self.episodes.append(traj)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def __len__(self) -> int:
        return len(self.episodes)

    def sample_subsequence(
        self, batch_size: int, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert len(self.episodes) > 0, "No data to sample from."
        B, T = int(batch_size), int(seq_len)

        obs_b = []
        prev_act_b = []
        rew_b = []
        act_b = []
        t_b = []

        for _ in range(B):
            traj = random.choice(self.episodes)
            L = traj.obs.shape[0]
            if L >= T:
                start = random.randint(0, L - T)
                sl = slice(start, start + T)
                obs = traj.obs[sl]
                act = traj.act[sl]
                rew = traj.rew[sl]
                tt = traj.t[sl]
            else:
                # pad by repeating last element (simple + deterministic)
                pad = T - L
                obs = np.concatenate(
                    [traj.obs, np.repeat(traj.obs[-1:], pad, axis=0)], axis=0
                )
                act = np.concatenate(
                    [traj.act, np.repeat(traj.act[-1:], pad, axis=0)], axis=0
                )
                rew = np.concatenate(
                    [traj.rew, np.repeat(traj.rew[-1:], pad, axis=0)], axis=0
                )
                tt = np.concatenate(
                    [traj.t, np.arange(traj.t[-1] + 1, traj.t[-1] + 1 + pad)], axis=0
                )

            prev_act = np.zeros_like(act)
            prev_act[1:] = act[:-1]

            obs_b.append(obs)
            prev_act_b.append(prev_act)
            rew_b.append(rew)
            act_b.append(act)
            t_b.append(tt)

        return (
            np.stack(obs_b, axis=0),  # [B,T,obs]
            np.stack(prev_act_b, axis=0),  # [B,T,act]
            np.stack(rew_b, axis=0),  # [B,T,1]
            np.stack(act_b, axis=0),  # [B,T,act] (target)
            np.stack(t_b, axis=0),  # [B,T]
        )


@torch.no_grad()
def policy_action(
    model: TrajectoryTransformerPolicy,
    device: torch.device,
    obs_hist: List[np.ndarray],
    act_hist: List[np.ndarray],
    rew_hist: List[float],
    action_low: np.ndarray,
    action_high: np.ndarray,
    max_ctx: int,
    deterministic: bool,
    exploration_std: float,
) -> np.ndarray:
    # obs_hist is always one step longer than act_hist (obs_0 then actions produce obs_1, ...)
    # Build a context of length L where all modalities align on time dim.
    L = min(int(max_ctx), len(obs_hist))
    act_dim = int(action_low.shape[0])

    obs_arr = np.stack(obs_hist[-L:], axis=0).astype(np.float32)  # [L, obs_dim]
    rew_arr = np.array(rew_hist[-L:], dtype=np.float32).reshape(L, 1)  # [L, 1]

    prev_act = np.zeros((L, act_dim), dtype=np.float32)  # [L, act_dim]
    if L > 1 and len(act_hist) > 0:
        # Take the last (L-1) actions to fill prev_act[1:].
        a_slice = act_hist[-(L - 1) :]
        prev_act[1:] = np.stack(a_slice, axis=0).astype(np.float32)

    # Timestep indices are local within the context window.
    t_arr = np.arange(L, dtype=np.int64)

    obs_t = torch.from_numpy(obs_arr).unsqueeze(0).to(device)
    prev_act_t = torch.from_numpy(prev_act).unsqueeze(0).to(device)
    rew_t = torch.from_numpy(rew_arr).unsqueeze(0).to(device)
    tt = torch.from_numpy(t_arr).unsqueeze(0).to(device)

    pred = model(obs_t, prev_act_t, rew_t, tt)[0, -1]  # [act_dim]
    a = pred.cpu().numpy()

    if not deterministic and exploration_std > 0:
        a = a + exploration_std * np.random.randn(*a.shape).astype(np.float32)

    return np.clip(a, action_low, action_high).astype(np.float32)


def collect_trajectory(
    env_name: str,
    model: Optional[TrajectoryTransformerPolicy],
    device: torch.device,
    action_low: np.ndarray,
    action_high: np.ndarray,
    max_steps: int,
    max_ctx: int,
    deterministic: bool,
    exploration_std: float,
) -> Tuple[Trajectory, Dict[str, float]]:
    env = make_env(env_name)
    obs, _ = env.reset()
    o = flatten_observation(obs)

    obs_hist: List[np.ndarray] = [o]
    act_hist: List[np.ndarray] = []
    rew_hist: List[float] = [0.0]

    obs_list, act_list, rew_list, done_list, t_list = [], [], [], [], []

    ep_ret = 0.0
    for t in range(int(max_steps)):
        if model is None:
            a = np.random.uniform(action_low, action_high).astype(np.float32)
        else:
            a = policy_action(
                model=model,
                device=device,
                obs_hist=obs_hist,
                act_hist=act_hist,
                rew_hist=rew_hist,
                action_low=action_low,
                action_high=action_high,
                max_ctx=max_ctx,
                deterministic=deterministic,
                exploration_std=exploration_std,
            )

        next_obs, r, terminated, truncated, _ = env.step(a)
        done = bool(terminated or truncated)
        no = flatten_observation(next_obs)

        obs_list.append(o)
        act_list.append(a)
        rew_list.append([float(r)])
        done_list.append([1.0 if done else 0.0])
        t_list.append(t)

        ep_ret += float(r)

        o = no
        obs_hist.append(o)
        act_hist.append(a)
        rew_hist.append(float(r))

        if done:
            break

    env.close()
    traj = Trajectory(
        obs=np.asarray(obs_list, dtype=np.float32),
        act=np.asarray(act_list, dtype=np.float32),
        rew=np.asarray(rew_list, dtype=np.float32),
        done=np.asarray(done_list, dtype=np.float32),
        t=np.asarray(t_list, dtype=np.int64),
    )
    info = {"episode_return": ep_ret, "episode_length": float(traj.obs.shape[0])}
    return traj, info


def train_transformer(
    env_name: str,
    total_env_steps: int,
    warmup_episodes: int,
    steps_per_iter: int,
    updates_per_iter: int,
    batch_size: int,
    seq_len: int,
    max_ctx: int,
    lr: float,
    seed: int,
    eval_every_iters: int,
    eval_episodes: int,
    exploration_std: float,
    log_dir: str,
    d_model: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim, action_dim, action_low, action_high = get_env_specs(env_name)

    model = TrajectoryTransformerPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        max_timestep=10000,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    buf = EpisodeBuffer(max_episodes=2000)

    # Warmup with random data so the first batches are well-formed.
    while len(buf) < int(warmup_episodes):
        traj, info = collect_trajectory(
            env_name=env_name,
            model=None,
            device=device,
            action_low=action_low,
            action_high=action_high,
            max_steps=1000,
            max_ctx=max_ctx,
            deterministic=False,
            exploration_std=exploration_std,
        )
        if traj.obs.shape[0] > 1:
            buf.add(traj)

    global_steps = 0
    it = 0
    while global_steps < int(total_env_steps):
        it += 1

        # 1) Collect on-policy-ish episodes/steps using current transformer.
        steps_collected = 0
        while steps_collected < int(steps_per_iter) and global_steps < int(
            total_env_steps
        ):
            traj, info = collect_trajectory(
                env_name=env_name,
                model=model,
                device=device,
                action_low=action_low,
                action_high=action_high,
                max_steps=min(1000, int(steps_per_iter - steps_collected)),
                max_ctx=max_ctx,
                deterministic=False,
                exploration_std=exploration_std,
            )
            L = int(traj.obs.shape[0])
            if L > 1:
                buf.add(traj)
                steps_collected += L
                global_steps += L
                wandb.log(
                    {
                        "collect/episode_return": info["episode_return"],
                        "collect/episode_length": info["episode_length"],
                        "env/global_steps": global_steps,
                        "data/episodes": len(buf),
                    }
                )
            else:
                # Avoid filling buffer with degenerate episodes.
                global_steps += max(1, L)

        # 2) Supervised updates from subsequences.
        model.train()
        losses = []
        for u in range(int(updates_per_iter)):
            obs_b, prev_act_b, rew_b, act_tgt_b, t_b = buf.sample_subsequence(
                batch_size=batch_size, seq_len=seq_len
            )
            obs = torch.from_numpy(obs_b).to(device)
            prev_act = torch.from_numpy(prev_act_b).to(device)
            rew = torch.from_numpy(rew_b).to(device)
            act_tgt = torch.from_numpy(act_tgt_b).to(device)
            tt = torch.from_numpy(t_b).to(device)

            pred = model(obs, prev_act, rew, tt)
            loss = F.mse_loss(pred, act_tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        wandb.log(
            {
                "train/loss_mse": float(np.mean(losses)) if losses else math.nan,
                "env/global_steps": global_steps,
                "iter": it,
            }
        )

        # 3) Periodic evaluation (deterministic, no exploration).
        if eval_every_iters > 0 and (it % int(eval_every_iters) == 0):
            model.eval()
            rets = []
            lens = []
            for _ in range(int(eval_episodes)):
                traj, info = collect_trajectory(
                    env_name=env_name,
                    model=model,
                    device=device,
                    action_low=action_low,
                    action_high=action_high,
                    max_steps=1000,
                    max_ctx=max_ctx,
                    deterministic=True,
                    exploration_std=0.0,
                )
                rets.append(info["episode_return"])
                lens.append(info["episode_length"])

            wandb.log(
                {
                    "eval/return_mean": float(np.mean(rets)) if rets else math.nan,
                    "eval/return_std": float(np.std(rets)) if rets else math.nan,
                    "eval/len_mean": float(np.mean(lens)) if lens else math.nan,
                    "env/global_steps": global_steps,
                }
            )

    # Save checkpoint
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"transformer_ckpt_{int(time.time())}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "action_low": action_low,
            "action_high": action_high,
            "seed": seed,
        },
        ckpt_path,
    )
    print(f"Saved transformer checkpoint to {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", type=str, default="cartpole::balance")

    p.add_argument("--total_env_steps", type=int, default=1_000_000)
    p.add_argument("--warmup_episodes", type=int, default=10)
    p.add_argument("--steps_per_iter", type=int, default=5000)
    p.add_argument("--updates_per_iter", type=int, default=1000)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--max_ctx", type=int, default=32)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--eval_every_iters", type=int, default=10)
    p.add_argument("--eval_episodes", type=int, default=3)

    p.add_argument("--exploration_std", type=float, default=0.1)

    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--wandb_project", type=str, default="mpo_project")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--base_log_dir", type=str, default="./logs/transformer_experiment")
    p.add_argument("--wandb_group_prefix", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed = args.seed if args.seed != 0 else (int(time.time()) % 10000)

    experiment_identifier = f"{args.env_name}__seed{seed}_" + time.strftime(
        "%Y%m%d-%H%M%S"
    )
    log_dir = os.path.join(
        args.base_log_dir, args.wandb_project + "_" + experiment_identifier
    )
    os.makedirs(log_dir, exist_ok=True)

    wandb.init(
        name=experiment_identifier,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=(
            f"{args.wandb_group_prefix}_tx_{args.env_name}"
            if args.wandb_group_prefix
            else f"tx_{args.env_name}"
        ),
        config=vars(args) | {"seed": seed},
        dir=log_dir,
    )

    train_transformer(
        env_name=args.env_name,
        total_env_steps=args.total_env_steps,
        warmup_episodes=args.warmup_episodes,
        steps_per_iter=args.steps_per_iter,
        updates_per_iter=args.updates_per_iter,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_ctx=args.max_ctx,
        lr=args.lr,
        seed=seed,
        eval_every_iters=args.eval_every_iters,
        eval_episodes=args.eval_episodes,
        exploration_std=args.exploration_std,
        log_dir=log_dir,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )

    wandb.finish()
