# PyTorch Scratch Experiments

Scratch and ablation implementations for MPO/V-MPO style training on Atari and dm_control.
These scripts are useful for smaller experiments and algorithm sanity checks.

## Files

- `train_dm_control_simple.py`: V-MPO style training on dm_control.
- `train_dm_control_popart.py`: dm_control variant with PopArt normalization.
- `train_atari_simple.py`: Atari V-MPO baseline.
- `train_atari_batch.py`: batched Atari V-MPO variant.
- `train_gae.py`: compact GAE-based V-MPO prototype.
- `train_q.py`: Q/advantage-style V-MPO prototype.
- `video_dm_control_simple.py`: video renderer for `train_dm_control_simple.py` checkpoints.
- `utils.py`: helper functions.
- `requirements.txt`: pinned dependencies for this folder.

## Run examples

Install local requirements:

```bash
pip install -r src/pytorch_scratch/requirements.txt
```

dm_control:

```bash
python src/pytorch_scratch/train_dm_control_simple.py --domain cheetah --task run
python src/pytorch_scratch/train_dm_control_popart.py --domain cheetah --task run
```

Atari:

```bash
python src/pytorch_scratch/train_atari_simple.py --game Pong
python src/pytorch_scratch/train_atari_batch.py --game Pong
```

Minimal prototypes:

```bash
python src/pytorch_scratch/train_gae.py
python src/pytorch_scratch/train_q.py
```

Video generation:

```bash
python src/pytorch_scratch/video_dm_control_simple.py \
  --checkpoint logs/dm_control_vmpo/checkpoints/ckpt.pt \
  --domain cheetah \
  --task run \
  --out videos/rollout.mp4
```
