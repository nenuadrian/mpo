import argparse
import os

import numpy as np
import torch
import gymnasium
import imageio

from gaussian_policy import GaussianPolicy
from util import load_policy_from_checkpoint, make_offscreen_env, generate_video

def main():
    parser = argparse.ArgumentParser(description="Generate video from MPO checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--env_name", type=str, default="HalfCheetah-v5", help="Gymnasium env name"
    )
    parser.add_argument(
        "--output", type=str, default="video.mp4", help="Output video path"
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to record"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Max steps per episode"
    )
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use policy mean deterministically instead of sampling",
    )
    parser.add_argument("--device", type=str, default="cpu", help="torch device")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    env = make_offscreen_env(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    policy = GaussianPolicy(obs_dim, act_dim, action_low, action_high)
    print(f"Loading policy from checkpoint: {ckpt_path}")
    load_policy_from_checkpoint(ckpt_path, policy)
    policy.to(args.device)
    policy.eval()

    generate_video(
        env_name=args.env_name,
        env=env,
        policy=policy,
        output_path=args.output,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        fps=args.fps,
        deterministic=args.deterministic,
        device=args.device,
    )


if __name__ == "__main__":
    main()
