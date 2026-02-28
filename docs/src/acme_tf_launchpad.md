# ACME TensorFlow MPO (Launchpad)

Single-file TensorFlow MPO setup based on DeepMind Acme and Launchpad, with additional DM dependencies (`dm-reverb`, `dm-launchpad`, `dm-control`, `dm-tree`, `sonnet`, `acme`).
This corresponds to section **I - Single file Acme TF with Launchpad** in the root report.

## Main entry points

- `train.py`: distributed Launchpad training program.
- `visualize_acme_results.py`: plots multiple CSV metrics into a single figure.

## Run

```bash
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/adrian_nenu/miniconda3/envs/acme/lib/
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python src/acme_tf_launchpad/train.py \
  --max_actor_steps 3000000 \
  --domain cheetah \
  --task run
```

## Result plots (from root report)

```bash
python src/acme_tf_launchpad/visualize_acme_results.py \
  --output results/acme_tf.png \
  --metric episode_return \
  --file "Cartpole_Balance::results/acme/mpo-tf-cartpole-balance.csv" \
  --file "Hopper_Stand::results/acme/mpo-tf-hopper-stand.csv" \
  --file "Hopper_Hop::results/acme/mpo-tf-hopper-hop.csv" \
  --file "Walker_Walk::results/acme/mpo-tf-walker-walk.csv" \
  --file "Walker_Run::results/acme/mpo-tf-walker-run.csv" \
  --file "Walker_Stand::results/acme/mpo-tf-walker-stand.csv" \
  --file "Reacher_Easy::results/acme/mpo-tf-reacher-easy.csv" \
  --file "Reacher_Hard::results/acme/mpo-tf-reacher-hard.csv" \
  --file "Acrobot_Swingup::results/acme/mpo-tf-acrobot-swingup.csv" \
  --file "Pendulum_Swingup::results/acme/mpo-tf-pendulum-swingup.csv" \
  --file "Swimmer_Swimmer6::results/acme/mpo-tf-swimmer-swimmer6.csv" \
  --file "Swimmer_Swimmer15::results/acme/mpo-tf-swimmer-swimmer15.csv" \
  --file "Cheetah_Run::results/acme/mpo-tf-cheetah-run.csv"
```
