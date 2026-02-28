# PyTorch MPO (Custom From Scratch)

Experimental custom MPO implementation in PyTorch.
This folder corresponds to section **V - Implementation from scratch in PyTorch (not working)** in the root report.

## Main entry points

- `train.py`: custom MPO trainer (Gymnasium tasks).
- `generate_video_custom_pytorch_mpo.py`: render a trained checkpoint to video.

## Run

```bash
python src/pytorch_custom/train.py \
  --env_name HalfCheetah-v5 \
  --max_actor_steps 1500000 \
  --wandb_project custom_mpo9
```

## Video generation

```bash
python src/pytorch_custom/generate_video_custom_pytorch_mpo.py \
  logs/mpo_experiment/checkpoints/checkpoint_ep309.pt \
  --env_name HalfCheetah-v5 \
  --output mpo_halfcheetah.mp4
```

## Notes

- This code path is kept for research iteration and comparison.
- The mainline stable PyTorch implementations in this repo are under `acme_pytorch*`, `acme_vmpo_pytorch`, and `mog_pytorch`.
