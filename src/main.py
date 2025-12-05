import argparse
import os
import json
import time

import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

from mpo.algorithm import train_mpo
from mpo.mpo_config import MPOConfig
from mpo.utils import generate_video


def main():
    parser = argparse.ArgumentParser(description="Train MPO")
    parser.add_argument(
        "--env_names",
        type=str,
        default="HalfCheetah-v5",
        help="Comma-separated list of environment names to train on",
    )
    parser.add_argument("--env_iterations", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_training_episodes", type=int, default=1000)
    parser.add_argument("--num_candidate_actions", type=int, default=32)
    parser.add_argument("--min_replay_size", type=int, default=30_000)
    parser.add_argument("--num_optimization_steps_per_step", type=int, default=2)
    parser.add_argument("--q_lr", type=float, default=0.0005)
    parser.add_argument("--pi_lr", type=float, default=0.0005)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--dual_lr", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--kl_epsilon", type=float, default=0.2)
    parser.add_argument("--policy_old_sync_frequency", type=int, default=50)
    parser.add_argument("--base_log_dir", type=str, default="./logs/mpo_experiment")
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--static_seed", type=int, default=None)
    parser.add_argument("--entropy_coeff", type=float, default=1e-3)
    parser.add_argument("--checkpoint_ep_freq", type=int, default=50)
    parser.add_argument("--wandb_project", type=str, default="mpo_project")
    parser.add_argument("--wandb_entity", type=str, default="adrian-research")
    parser.add_argument("--wandb_group_prefix", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    env_names = [name.strip() for name in args.env_names.split(",")]
    for env_name in env_names:
        for iteration in range(args.env_iterations):
            print(
                f"Training on environment: {env_name}. Starting iteration {iteration + 1}/{args.env_iterations}"
            )
            start_time = time.time()

            seed = (
                torch.randint(0, 10000, (1,)).item()
                if args.static_seed is None
                else args.static_seed
            )

            experiment_identifier = (
                env_name
                + "_iter"
                + str(iteration + 1)
                + f"_seed{seed}"
                + "_"
                + time.strftime("%Y%m%d-%H%M%S")
            )

            config = MPOConfig(
                env_name=env_name,
                seed=seed,
                log_dir=os.path.join(
                    args.base_log_dir, args.wandb_project + "_" + experiment_identifier
                ),
                **vars(args),
            )
            os.makedirs(config.log_dir, exist_ok=True)

            print("Experiment Configuration:")
            print(json.dumps(vars(config), indent=4))

            with open(os.path.join(config.log_dir, "config.json"), "w") as f:
                json.dump(vars(config), f, indent=4)

            wandb.init(
                name=experiment_identifier,
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=(
                    f"{args.wandb_group_prefix}_mpo_{env_name}"
                    if args.wandb_group_prefix
                    else f"mpo_{env_name}"
                ),
                config=vars(args),
                sync_tensorboard=True,
            )
            writer = SummaryWriter(os.path.join(config.log_dir, "tb_logs"))

            pi = train_mpo(config, device, writer)

            try:
                generate_video(
                    env_name=env_name,
                    policy=pi,
                    output_path=os.path.join(config.log_dir, "video.mp4"),
                    num_episodes=2,
                    max_steps=1000,
                    fps=30,
                    deterministic=False,
                    device=device,
                )
            except Exception as e:
                print(f"[ERROR] Warning: video generation failed: {e}")

            writer.close()
            wandb.finish()
            end_time = time.time()
            print(
                f"Iteration {iteration + 1}/{args.env_iterations} completed in "
                f"{end_time - start_time:.2f} seconds."
            )


if __name__ == "__main__":
    main()
