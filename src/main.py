import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

from algorithm import train_mpo, MPOConfig


def main():
    parser = argparse.ArgumentParser(description="Train MPO")
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v5")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_training_episodes", type=int, default=1000)
    parser.add_argument("--num_candidate_actions", type=int, default=4)
    parser.add_argument("--min_replay_size", type=int, default=100)
    parser.add_argument("--num_optimization_steps_per_step", type=int, default=2)
    parser.add_argument("--q_lr", type=float, default=0.0005)
    parser.add_argument("--pi_lr", type=float, default=0.0005)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--dual_lr", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--kl_epsilon", type=float, default=0.1)
    parser.add_argument("--policy_old_sync_frequency", type=int, default=50)
    parser.add_argument("--log_dir", type=str, default="./logs/mpo_experiment")
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entropy_coeff", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = MPOConfig(**vars(args))

    print("Experiment Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    wandb.init(project="mpo_project", config=vars(args), sync_tensorboard=True)
    writer = SummaryWriter(config.log_dir)

    train_mpo(config, device, writer)

    writer.close()
    wandb.finish()


if __name__ == "__main__":
    main()
