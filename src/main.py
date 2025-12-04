import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

from algorithm import train_mpo, MPOConfig


def main():
    args = argparse.ArgumentParser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = MPOConfig(
        batch_size=64,
        num_training_episodes=1000,
        num_candidate_actions=4,
        min_replay_size=100,
        num_optimization_steps=1000,
        q_lr=0.0005,
        pi_lr=0.0005,
        tau=0.005,
        dual_lr=1e-3,
        eta=1.0,
        kl_epsilon=0.1,
        policy_old_sync_frequency=50,
        log_dir="./logs/mpo_experiment",
        eval_freq=10,
        eval_episodes=5,
        seed=42,
        entropy_coeff=1e-3,
    )

    checkpoint_dir = os.path.join(config.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb.init(
        project="mpo_project",
        config={
            "batch_size": config.batch_size,
            "num_training_episodes": config.num_training_episodes,
            "num_candidate_actions": config.num_candidate_actions,
            "min_replay_size": config.min_replay_size,
            "num_optimization_steps": config.num_optimization_steps,
            "q_lr": config.q_lr,
            "pi_lr": config.pi_lr,
            "tau": config.tau,
            "dual_lr": config.dual_lr,
            "kl_epsilon": config.kl_epsilon,
            "seed": config.seed,
            "eval_freq": config.eval_freq,
            "eval_episodes": config.eval_episodes,
            "entropy_coeff": config.entropy_coeff,
        },
        sync_tensorboard=True,
    )
    writer = SummaryWriter(config.log_dir)

    train_mpo(config, device, writer)

    writer.close()
    wandb.finish()


if __name__ == "__main__":
    main()
