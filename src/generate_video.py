import argparse
import os

import numpy as np
import torch
import gymnasium
import imageio

from gaussian_policy import GaussianPolicy


def _make_offscreen_env(env_name: str):
    """
    Try to create a gymnasium MuJoCo env with an offscreen GL backend.
    Tries MUJOCO_GL in order: 'egl', 'osmesa', 'glfw'. Raises RuntimeError
    with diagnostic if none work.
    """
    import gymnasium

    last_err = None
    for backend in ("egl", "osmesa", "glfw"):
        os.environ["MUJOCO_GL"] = backend
        try:
            env = gymnasium.make(env_name, render_mode="rgb_array")
            # Try a quick reset + render to ensure backend works
            try:
                obs, _ = env.reset()
                frame = env.render()
            except Exception:
                # some envs only create context later; treat make as success
                pass
            print(f"[generate_video] using MUJOCO_GL={backend}")
            return env
        except Exception as e:
            last_err = e
            print(f"[generate_video] backend {backend} failed: {e}")
            try:
                env.close()
            except Exception:
                pass
    raise RuntimeError(
        "Failed to initialize an offscreen MuJoCo OpenGL context. "
        "Tried MUJOCO_GL backends 'egl','osmesa','glfw'.\n"
        "Last error: "
        + str(last_err)
        + "\nInstall/configure EGL or OSMesa (or run with a display) and retry."
    )


def load_policy_from_checkpoint(ckpt_path: str, policy: torch.nn.Module):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Prefer 'pi_state_dict' then 'pi_old_state_dict'
    if "pi_state_dict" in ckpt:
        state = ckpt["pi_state_dict"]
    elif "pi_old_state_dict" in ckpt:
        state = ckpt["pi_old_state_dict"]
    else:
        # attempt to use top-level state_dict if present
        state = ckpt.get("state_dict", ckpt)
    policy.load_state_dict(state)
    return ckpt


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

    # Create env that returns rgb frames. Use helper that tries offscreen backends.
    env = _make_offscreen_env(args.env_name)
    obs0, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    policy = GaussianPolicy(obs_dim, act_dim, action_low, action_high)
    print(f"Loading policy from checkpoint: {ckpt_path}")
    load_policy_from_checkpoint(ckpt_path, policy)
    policy.to(args.device)
    policy.eval()

    writer = imageio.get_writer(args.output, fps=args.fps)

    try:
        print(f"Recording {args.episodes} episodes to {args.output}...")
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            # render initial frame (some envs require rendering after reset)
            try:
                frame = env.render()
            except Exception as e:
                frame = None
                print(f"[generate_video] warning: initial render failed: {e}")
            if frame is not None:
                # ensure uint8
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                writer.append_data(frame)

            while not done and steps < args.max_steps:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=args.device
                ).unsqueeze(0)
                with torch.no_grad():
                    if args.deterministic:
                        mu, _ = policy(obs_t)
                        a_tanh = torch.tanh(mu)
                        action_t = (
                            (a_tanh * policy.action_scale + policy.action_bias)
                            .cpu()
                            .numpy()[0]
                        )
                    else:
                        action_t, _ = policy.sample(obs_t)
                        action_t = action_t.cpu().numpy()[0]

                next_obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated
                obs = next_obs

                try:
                    frame = env.render()
                except Exception as e:
                    frame = None
                    print(
                        f"[generate_video] warning: render failed at step {steps}: {e}"
                    )

                if frame is not None:
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                    writer.append_data(frame)

                steps += 1

            print(f"Episode {ep+1} recorded, steps={steps}")

    finally:
        writer.close()
        env.close()

    print(f"Saved video to: {args.output}")


if __name__ == "__main__":
    main()
