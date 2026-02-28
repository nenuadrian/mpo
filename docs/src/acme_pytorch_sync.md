# ACME PyTorch Sync (MPO)

Single-threaded PyTorch MPO implementation based on the ACME-style PyTorch learner.
This folder corresponds to section **IV - Single-threaded Implementation of III** in the root report.

## Main entry points

- `train.py`: single-threaded MPO trainer for dm_control tasks.
- `train_bulk.sh`: convenience script to run multiple `domain::task` pairs.

## Run

```bash
python src/acme_pytorch_sync/train.py \
  --env_name cheetah::run \
  --max_actor_steps 3000000 \
  --wandb_project acme_pytorch_mpo_sync
```

Bulk run:

```bash
bash src/acme_pytorch_sync/train_bulk.sh
```

## Result plots (from root report)

Without action penalization:

```bash
python src/visualize_wandb.py \
  --project-metric "acme_pytorch_mpo_sync_no_action_penalization::eval/episode_return" \
  --entity "adrian-research" \
  --cache-dir logs \
  --show-individual \
  --output results/custom_acme_pytorch_sync_no_ap.png \
  --ncols 2
```

Without action penalization and per-dimension constraining:

```bash
python src/visualize_wandb.py \
  --project-metric "acme_pytorch_mpo_sync_no_AP_DC::eval/episode_return" \
  --entity "adrian-research" \
  --cache-dir logs \
  --show-individual \
  --output results/custom_acme_pytorch_sync_no_ap_dc.png \
  --ncols 2
```
