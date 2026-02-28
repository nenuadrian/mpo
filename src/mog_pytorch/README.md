# ACME PyTorch Sync (MoG MPO)

PyTorch Mixture-of-Gaussians MPO variant built on top of the synchronized ACME-style MPO implementation.
This corresponds to section **VII - MoG MPO based on IV** in the root report.

## Main entry points

- `train.py`: MoG-MPO training loop for dm_control tasks.
- `train_bulk.sh`: convenience script for benchmark task batches.

## Run

```bash
python src/mog_pytorch/train.py \
  --env_name cheetah::run \
  --max_actor_steps 3000000 \
  --wandb_project mog_pytorch_sync
```

Bulk run:

```bash
bash src/mog_pytorch/train_bulk.sh
```

## Result plots (from root report)

```bash
python src/visualize_wandb.py \
  --project-metric "mog_pytorch_sync::eval/episode_return" \
  --entity "adrian-research" \
  --cache-dir logs \
  --show-individual \
  --output results/mog_pytorch.png
```
