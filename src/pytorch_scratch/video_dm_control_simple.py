import argparse
import os

import gymnasium as gym
import numpy as np
import shimmy  # noqa: F401
import torch
import imageio

from train_dm_control_simple import GaussianPolicy

from utils import set_seed, make_dm_control_env, flatten_obs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def render_policy(
    checkpoint_path: str,
    domain: str,
    task: str,
    out_path: str,
    seed: int,
    max_steps: int,
    fps: int,
):
    set_seed(seed)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # --- Build env with RGB rendering ---
    env = make_dm_control_env(
        domain,
        task,
        seed=seed,
        render_mode="rgb_array",
    )

    # --- Infer dimensions ---
    if isinstance(env.observation_space, gym.spaces.Dict):
        obs_dim = sum(
            int(np.prod(v.shape)) for v in env.observation_space.spaces.values()
        )
    else:
        obs_dim = int(np.prod(env.observation_space.shape))

    act_dim = int(np.prod(env.action_space.shape))

    # --- Build policy and load weights ---
    policy = GaussianPolicy(obs_dim, act_dim).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    # --- Rollout ---
    frames = []

    obs, _ = env.reset()
    obs = flatten_obs(obs)

    for t in range(max_steps):
        frame = env.render()
        frames.append(frame)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action = policy.act_deterministic(obs_t).cpu().numpy().squeeze(0)

        action = np.clip(
            action,
            env.action_space.low,
            env.action_space.high,
        )

        obs, _, terminated, truncated, _ = env.step(action)
        obs = flatten_obs(obs)

        if terminated or truncated:
            break

    env.close()

    # --- Write video ---
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)

    print(f"Saved video to: {out_path}")
    print(f"Frames: {len(frames)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a trained V-MPO checkpoint to video"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--out", type=str, default="videos/rollout.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    render_policy(
        checkpoint_path=args.checkpoint,
        domain=args.domain,
        task=args.task,
        out_path=args.out,
        seed=args.seed,
        max_steps=args.max_steps,
        fps=args.fps,
    )
