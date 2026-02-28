# ACME PyTorch Sync (V-MPO)

PyTorch V-MPO variant built on top of the synchronized ACME-style MPO implementation.
This corresponds to section **VI - V-MPO based on IV** in the root report.

## Main entry points

- `train.py`: V-MPO training loop for dm_control tasks.
- `train_bulk.sh`: convenience script for benchmark task batches.

## Run

```bash
python src/acme_vmpo_pytorch/train.py \
  --env_name cheetah::run \
  --max_actor_steps 3000000 \
  --wandb_project acme_pytorch_vmpo_sync
```

Bulk run:

```bash
bash src/acme_vmpo_pytorch/train_bulk.sh
```

## Result plots (from root report)

```bash
python src/visualize_wandb.py \
  --project-metric "acme_pytorch_vmpo_sync::eval/episode_return" \
  --entity "adrian-research" \
  --cache-dir logs \
  --show-individual \
  --output results/acme_vmpo_pytorch.png
```
