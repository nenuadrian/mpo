# ACME TensorFlow MPO (Reduced Dependencies)

Single-file TensorFlow MPO implementation derived from the Launchpad version, reduced to core dependencies around `dm-reverb` and `dm-sonnet`.
This corresponds to section **II - Single file Acme TF based on I with only dm-Reverb and dm-Sonnet** in the root report.

## Main entry points

- `train.py`: reduced-dependency MPO trainer.
- `train_acme_custom.sh`: convenience multi-task launcher.
- `reverb_base.py`, `reverb_transition.py`: replay definitions.

## Run

```bash
python src/acme_tf_reduced_deps/train.py \
  --max_actor_steps 3000000 \
  --domain cheetah \
  --task run
```

Bulk run:

```bash
bash src/acme_tf_reduced_deps/train_acme_custom.sh
```

## Result plots (from root report)

```bash
python src/visualize_wandb.py \
  --project-metric "G_ACME_TF_CUSTOM::evaluator/episode_return" \
  --entity "adrian-research" \
  --cache-dir logs \
  --show-individual \
  --output results/custom_acme_tf.png \
  --ncols 2
```
