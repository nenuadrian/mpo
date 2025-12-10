import argparse
import copy
import math
import numpy as np
import os
import torch
import gymnasium as gym
import cv2
from typing import Tuple

# Import necessary classes from the training file
from train_custom_acme_pytorch_mpo import MPOAgent, make_env, flatten_observation


def load_agent_from_checkpoint(
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    device: torch.device,
    **agent_kwargs,
) -> MPOAgent:
    """Load MPOAgent from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create agent with the same parameters
    agent = MPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        device=device,
        lr_dual=1,
        min_replay_size=10,
        **agent_kwargs,
    )

    # Load state dicts
    agent.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
    agent.policy_head.load_state_dict(checkpoint["policy_head"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.target_obs_encoder.load_state_dict(checkpoint["target_obs_encoder"])
    agent.target_policy_head.load_state_dict(checkpoint["target_policy_head"])
    agent.target_critic.load_state_dict(checkpoint["target_critic"])
    agent.mpo_loss.load_state_dict(checkpoint["mpo_loss"])
    agent.critic_opt.load_state_dict(checkpoint["critic_opt"])
    agent.policy_opt.load_state_dict(checkpoint["policy_opt"])
    agent.dual_opt.load_state_dict(checkpoint["dual_opt"])
    agent._learn_steps = checkpoint["learn_steps"]

    # Optionally load replay buffer if needed, but not required for inference
    # agent.replay._buffer = collections.deque([NStepTransition(*t) for t in checkpoint["replay"]], maxlen=agent.replay._max_size)

    agent.obs_encoder.eval()
    agent.policy_head.eval()
    agent.critic.eval()

    return agent


def generate_video(
    env_name: str,
    checkpoint_path: str,
    num_steps: int,
    output_path: str,
    fps: int = 30,
    render_mode: str = "rgb_array",
    render_kwargs: dict | None = None,
    **agent_kwargs,
):
    """Generate a video of the agent executing in the environment."""
    if render_kwargs is None:
        render_kwargs = {"height": 480, "width": 640}  # Default render size

    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Create environment
    env = make_env(env_name, render_mode=render_mode)

    # Get observation and action dimensions
    obs0, _ = env.reset()
    obs_flat = flatten_observation(obs0)
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Dict):
        obs_dim = sum(int(np.prod(sp.shape)) for sp in obs_space.spaces.values())
    else:
        obs_dim = int(np.prod(obs_space.shape))
    act_space = env.action_space
    action_dim = int(np.prod(act_space.shape))
    action_low = act_space.low
    action_high = act_space.high

    # Load agent
    agent = load_agent_from_checkpoint(
        checkpoint_path,
        obs_dim,
        action_dim,
        action_low,
        action_high,
        device,
        **agent_kwargs,
    )

    # Reset environment
    obs, _ = env.reset()
    obs_flat = flatten_observation(obs)

    frames = []
    step = 0
    done = False

    while step < num_steps and not done:
        # Select deterministic action
        action = agent.select_action(obs_flat, stochastic=False)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs_flat = flatten_observation(next_obs)
        step += 1

    env.close()

    # Save video using OpenCV
    if frames:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)
        video_writer.release()
        print(f"Video saved to {output_path}")
    else:
        print("No frames captured.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate video from MPO agent checkpoint."
    )
    parser.add_argument(
        "--env_name",
        type=str,
        required=True,
        help="Environment name, e.g., 'cartpole::balance'",
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument(
        "--num_steps", type=int, default=1000, help="Number of steps to run the agent"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="agent_video.mp4",
        help="Output video file path",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for the video"
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="rgb_array",
        choices=["rgb_array", "depth_array", "multi_camera"],
        help="Render mode",
    )
    parser.add_argument("--render_height", type=int, default=480, help="Render height")
    parser.add_argument("--render_width", type=int, default=640, help="Render width")
    # Agent kwargs can be added if needed, e.g., policy_hidden, etc., but defaults from MPOAgent
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_kwargs = {"height": args.render_height, "width": args.render_width}
    generate_video(
        env_name=args.env_name,
        checkpoint_path=args.checkpoint_path,
        num_steps=args.num_steps,
        output_path=args.output_path,
        fps=args.fps,
        render_mode=args.render_mode,
        render_kwargs=render_kwargs,
    )
