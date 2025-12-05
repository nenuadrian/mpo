import os
import sys

import torch
import numpy as np
import gymnasium
import imageio
import torch.nn as nn

from mpo.gaussian_policy import GaussianPolicy
from mpo.q_network import QNetwork
from mpo.mpo_config import MPOConfig


def compute_grad_stats(params):
    """Compute L2 grad norm and max-abs grad using torch (returns floats)."""
    total_sq = None
    max_abs = None
    found = False
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        g_sq = (g**2).sum()
        g_max = g.abs().max()
        if not found:
            total_sq = g_sq
            max_abs = g_max
            found = True
        else:
            total_sq = total_sq + g_sq
            max_abs = torch.max(max_abs, g_max)
    if not found:
        return {"grad_norm": 0.0, "grad_max": 0.0}
    grad_norm = torch.sqrt(total_sq).item()
    return {"grad_norm": float(grad_norm), "grad_max": float(max_abs.item())}


def evaluate_policy(policy: GaussianPolicy, env, device, n_eval_episodes: int = 5):
    """
    Run the policy for n_eval_episodes (stochastic sampling) and return list of episode returns and episode lengths.
    """
    returns = []
    lengths = []
    for _ in range(n_eval_episodes):
        obs = torch.tensor(
            env.reset()[0], dtype=torch.float32, device=device
        ).unsqueeze(0)
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            with torch.no_grad():
                action_tensor, _ = policy.sample(obs)
                action = action_tensor.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            done = terminated or truncated
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(
                0
            )
        returns.append(ep_ret)
        lengths.append(ep_len)
    return returns, lengths


def checkpoint_if_needed(
    config: MPOConfig,
    episode: int,
    global_step: int,
    q: QNetwork,
    q_target: QNetwork,
    pi: GaussianPolicy,
    pi_old: GaussianPolicy,
    q_optimizer: torch.optim.Optimizer,
    pi_optimizer: torch.optim.Optimizer,
) -> bool:
    if (episode + 1) % config.checkpoint_ep_freq != 0:
        return False

    checkpoint_dir = os.path.join(config.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        checkpoint = {
            "episode": episode + 1,
            "global_step": global_step,
            "q_state_dict": q.state_dict(),
            "q_target_state_dict": q_target.state_dict(),
            "pi_state_dict": pi.state_dict(),
            "pi_old_state_dict": pi_old.state_dict(),
            "q_optimizer_state_dict": q_optimizer.state_dict(),
            "pi_optimizer_state_dict": pi_optimizer.state_dict(),
        }
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode+1}.pt")
        torch.save(checkpoint, ckpt_path)
        torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint_latest.pt"))
    except Exception as e:
        print(f"[ERROR] failed to save checkpoint: {e}")
        return False
    return True


def make_offscreen_env(env_name: str):
    """
    Create a gymnasium MuJoCo env with an offscreen-capable backend,
    trying a small set of MUJOCO_GL configurations depending on OS.

    Linux:
        Try MUJOCO_GL in order: 'egl', 'osmesa', 'glfw'.

    macOS:
        First clear MUJOCO_GL (use default Metal/GLFW path),
        then try MUJOCO_GL='glfw' as a fallback.

    Raises RuntimeError with diagnostics if everything fails.
    """
    last_err = None
    platform = sys.platform

    if platform.startswith("linux"):
        backend_candidates = ("egl", "osmesa", "glfw")
    elif platform == "darwin":
        # None = "do not set MUJOCO_GL at all"
        backend_candidates = (None, "glfw")
    else:
        # best-effort generic fallback
        backend_candidates = (None, "egl", "osmesa", "glfw")

    for backend in backend_candidates:
        # Configure MUJOCO_GL for this attempt
        if backend is None:
            os.environ.pop("MUJOCO_GL", None)
            backend_name = "<default>"
        else:
            os.environ["MUJOCO_GL"] = backend
            backend_name = backend

        try:
            env = gymnasium.make(env_name, render_mode="rgb_array")

            # Some envs lazily create the context; force a reset + render
            try:
                obs, _ = env.reset()
                frame = env.render()
                # If the render returns None, just treat it as "not ready yet"
                _ = frame
            except Exception:
                # It's fine; env creation itself worked
                pass

            print(
                f"[make_offscreen_env] using MUJOCO_GL={backend_name} "
                f"on platform={platform}"
            )
            return env

        except Exception as e:
            last_err = e
            print(
                f"[make_offscreen_env] backend {backend_name} failed on "
                f"{platform}: {e}"
            )
            try:
                env.close()
            except Exception:
                pass

    raise RuntimeError(
        "Failed to initialize an offscreen MuJoCo rendering backend.\n"
        f"Platform: {platform}\n"
        f"Tried backends (in order): {backend_candidates}\n"
        f"Last error: {last_err}\n"
        "On Linux, ensure EGL/OSMesa are installed or try running with a display.\n"
        "On macOS, modern MuJoCo uses Metal; avoid forcing MUJOCO_GL to EGL/OSMesa."
    )


def generate_video(
    env_name: str,
    policy: torch.nn.Module,
    output_path: str,
    num_episodes: int = 1,
    max_steps: int = 1000,
    fps: int = 30,
    deterministic: bool = False,
    env: gymnasium.Env = None,
    device: str = "cpu",
):
    if env is None:
        env = make_offscreen_env(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    obs0, _ = env.reset()
    writer = imageio.get_writer(output_path, fps=fps)

    try:
        print(f"Recording {num_episodes} episodes to {output_path}...")
        for ep in range(num_episodes):
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

            while not done and steps < max_steps:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    if deterministic:
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

    print(f"Saved video to: {output_path}")


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
