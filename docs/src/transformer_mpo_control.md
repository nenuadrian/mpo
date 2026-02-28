# Transformer MPO Control

Trajectory-transformer policy for continuous-control experiments on dm_control with MPO-style training signals.

## Main entry point

- `train.py`: collects trajectories, trains a causal transformer policy, evaluates periodically, and saves checkpoints.

## Run

```bash
python src/transformer_mpo_control/train.py \
  --env_name cheetah::run \
  --total_env_steps 1000000 \
  --seq_len 32 \
  --wandb_project transformer_mpo_control
```

## Notes

- Defaults use dm_control task naming with `domain::task`.
- Logs and checkpoints are written under `--base_log_dir` (default: `./logs/transformer_experiment`).
