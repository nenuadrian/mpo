import argparse
import sys
from pathlib import Path
import os
import time

import numpy as np
import torch
import imageio.v2 as imageio

# allow importing the training module when running the script from anywhere
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_custom_acme_pytorch_mpo import (
    MPOAgent,
    make_env,
    flatten_observation,
)  # noqa: E402


def try_render(env):
    # Try a few common render calls (gymnasium / dm_control via shimmy).
    rgb = None
    try:
        rgb = env.render()
    except Exception:
        pass
    if rgb is None:
        try:
            rgb = env.render(mode="rgb_array")
        except Exception:
            rgb = None
    if rgb is None:
        try:
            # some envs (dm_control wrappers) expose render(height,width)
            rgb = env.render(width=640, height=480)
        except Exception:
            rgb = None
    return rgb


# New: normalize various render return shapes to (H,W,3) uint8 RGB.
def normalize_frame(frame):
    import numpy as _np
    from PIL import Image

    if frame is None:
        return None
    arr = _np.asarray(frame)

    # Reject obviously invalid frames
    if arr.size == 0:
        return None

    # If floats, assume range [0,1], clamp and scale
    if _np.issubdtype(arr.dtype, _np.floating):
        arr = _np.clip(arr, 0.0, 1.0)
        arr = (255.0 * arr).astype(_np.uint8)
    else:
        # convert integer types to uint8 if possible
        if arr.dtype != _np.uint8:
            try:
                arr = arr.astype(_np.uint8)
            except Exception:
                return None

    # First try the straightforward PIL conversion
    try:
        img = Image.fromarray(arr)
    except Exception:
        # Fallback: try to interpret flat / odd shapes as RGB pixel stream.
        total = arr.size
        if total % 3 != 0:
            return None
        pixels = total // 3

        img = None
        # Try a list of sensible (H,W) candidates where H*W == pixels.
        candidates = [(480, 640), (360, 640), (720, 1280), (256, 256), (224, 224)]
        for H, W in candidates:
            if H * W == pixels:
                try:
                    out = arr.reshape((H, W, 3))
                    img = Image.fromarray(out)
                    break
                except Exception:
                    continue

        if img is None:
            # Try common widths (so a (1,640,3) case will be handled).
            for W in (640, 320, 256, 128, 84):
                if pixels % W == 0:
                    H = pixels // W
                    try:
                        out = arr.reshape((H, W, 3))
                        img = Image.fromarray(out)
                        break
                    except Exception:
                        continue

        if img is None:
            # Last resort: reshape to (pixels,3) -> (1, pixels, 3) so PIL can make an image,
            # then allow resize to target resolution.
            try:
                out = arr.reshape((pixels, 3))
                out = out.reshape((1, pixels, 3))
                img = Image.fromarray(out)
            except Exception:
                return None

    try:
        img = img.convert("RGB")
        # Resize to (width, height) accepted by many video writers (640x480).
        img = img.resize((640, 480))
        out = _np.asarray(img, dtype=_np.uint8)
    except Exception:
        return None

    # Final sanity check
    if out.ndim != 3 or out.shape[2] != 3:
        return None

    return out


def load_agent_from_checkpoint(
    checkpoint_path: str, env_name: str, device: torch.device
):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    env = make_env(env_name)
    obs, _ = env.reset()
    obs_flat = flatten_observation(obs)
    obs_space = env.observation_space
    if isinstance(obs_space, dict) or hasattr(obs_space, "spaces"):
        obs_dim = sum(int(np.prod(sp.shape)) for sp in obs_space.spaces.values())
    else:
        obs_dim = int(np.prod(obs_space.shape))
    act_space = env.action_space
    action_dim = int(np.prod(act_space.shape))
    action_low = act_space.low
    action_high = act_space.high
    env.close()

    agent = MPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        device=device,
    )

    # Load weights if present in checkpoint
    def _maybe_load(obj, key):
        if key in ckpt and hasattr(obj, "load_state_dict"):
            try:
                obj.load_state_dict(ckpt[key])
            except Exception as e:
                print(f"Warning: failed to load {key}: {e}")

    _maybe_load(agent.obs_encoder, "obs_encoder")
    _maybe_load(agent.policy_head, "policy_head")
    _maybe_load(agent.critic, "critic")
    _maybe_load(agent.target_obs_encoder, "target_obs_encoder")
    _maybe_load(agent.target_policy_head, "target_policy_head")
    _maybe_load(agent.target_critic, "target_critic")
    _maybe_load(agent.mpo_loss, "mpo_loss")

    return agent


def record(
    checkpoint: str,
    env_name: str,
    out_path: str,
    episodes: int = 1,
    max_steps: int = 1000,
    fps: int = 30,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    agent = load_agent_from_checkpoint(checkpoint, env_name, device)

    env = make_env(env_name)

    # Create writer without passing the 'quality' kwarg which PyAVPlugin.write
    # doesn't accept in some imageio/pyav versions. Try to request the codec and
    # fall back to a plain writer for maximum compatibility.
    try:
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264")
    except TypeError:
        writer = imageio.get_writer(out_path, fps=fps)

    for ep in range(episodes):
        o, _ = env.reset()
        o_flat = flatten_observation(o)
        done = False
        steps = 0
        start = time.time()
        while not done and steps < max_steps:
            a = agent.select_action(o_flat, stochastic=False)
            # some environments expect actions shaped differently; use as-is
            no, r, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)
            # render frame
            frame = try_render(env)
            # Normalize render outputs to HxWx3 uint8; fall back to placeholder if
            # normalization fails (handles grayscale, RGBA, channel-first, floats).
            frame = normalize_frame(frame)
            if frame is None:
                # fallback: construct a small placeholder frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8) + 30
            # At this point 'frame' should be HxWx3 uint8 suitable for writer

            writer.append_data(frame)
            o_flat = flatten_observation(no)
            steps += 1
        print(f"Episode {ep+1}/{episodes} recorded, steps={steps}")
    writer.close()
    env.close()
    print(f"Saved video to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Path to checkpoint .pt saved by training")
    p.add_argument(
        "--env_name", required=True, help="Environment name (same used in training)"
    )
    p.add_argument("--output", required=True, help="Output video path (e.g. out.mp4)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(Path(args.output).parent, exist_ok=True)
    record(
        checkpoint=args.checkpoint,
        env_name=args.env_name,
        out_path=args.output,
        episodes=args.episodes,
        max_steps=args.max_steps,
        fps=args.fps,
    )
