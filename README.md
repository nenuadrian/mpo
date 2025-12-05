# MPO From Scratch

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install numpy torch matplotlib "gymnasium[mujoco]" tensorboard wandb

python src/main.py --seed 42 --num_training_episodes 10000
```
