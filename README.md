# On implementing Maximum a Posteriori Policy Optimization (MPO)

- [On implementing Maximum a Posteriori Policy Optimization (MPO)](#on-implementing-maximum-a-posteriori-policy-optimization-mpo)
  - [Implementing from scratch in PyTorch](#implementing-from-scratch-in-pytorch)
    - [Eval](#eval)
    - [Cheeting cheetah](#cheeting-cheetah)
    - [visualize](#visualize)
  - [Using Google DeepMind Acme](#using-google-deepmind-acme)
    - [visualise](#visualise)
  - [Implementing based on Acme in PyTorch](#implementing-based-on-acme-in-pytorch)
    - [visualize Acme in PyTorch](#visualize-acme-in-pytorch)
  - [Original MPO Paper](#original-mpo-paper)


## Implementing from scratch in PyTorch

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas torch matplotlib "gymnasium[mujoco]" tensorboard wandb "imageio[ffmpeg]"

python src/train_custom_pytorch_mpo.py --wandb_project custom_mpo9 --max_actor_steps 1500000 --env_names HalfCheetah-v5,Walker2d-v5 --env_iterations 1

python src/generate_video.py logs/mpo_experiment/checkpoints/checkpoint_ep309.pt --env_name HalfCheetah-v5 --output mpo_halfcheetah.mp4
```

### Eval

![MPO Training](assets/graph.png)

### Cheeting cheetah

![MPO Video](assets/cheetah.gif)

### visualize

```bash
python src/visualize_wandb.py --project-metric "ppo6::eval/mean_reward" --project-metric "custom_mpo6::eval/mean_reward" --entity "adrian-research" --cache-dir logs --show-individual --output results/custom_pytorch.png
```

![Custom Results](results/custom_pytorch.png)


## Using Google DeepMind Acme

Very difficult to get it to build. TensorFlow version is only one that I was able to get working.

Combined all code necessary in `src/train_acme_lp_tf_mpo.py`. But still need a lot of the tricks I have documented in my `acme` fork.

### visualise

```bash
python src/visualize_acme_results.py --output results/acme/metric_plot.png --metric episode_return --file "Cartpole_Balance::results/acme/mpo-tf-cartpole-balance.csv" --file "Hopper_Stand::results/acme/mpo-tf-hopper-stand.csv" --file "Hopper_Hop::results/acme/mpo-tf-hopper-hop.csv" --file "Walker_Walk::results/acme/mpo-tf-walker-walk.csv" --file "Walker_Run::results/acme/mpo-tf-walker-run.csv" --file "Walker_Stand::results/acme/mpo-tf-walker-stand.csv" --file "Reacher_Easy::results/acme/mpo-tf-reacher-easy.csv" --file "Reacher_Hard::results/acme/mpo-tf-reacher-hard.csv" --file "Reacher_Hard::results/acme/mpo-tf-reacher-hard.csv"  --file "Acrobot_Swingup::results/acme/mpo-tf-acrobot-swingup.csv"   --file "Pendulum_Swingup::results/acme/mpo-tf-pendulum-swingup.csv"  --file "Swimmer_Swimmer6::results/acme/mpo-tf-swimmer-swimmer6.csv" --file "Swimmer_Swimmer15::results/acme/mpo-tf-swimmer-swimmer15.csv" --file "Cheetah_Run::results/acme/mpo-tf-cheetah-run.csv" 
```

![Acme Results](results/acme/metric_plot.png)

## Implementing based on Acme in PyTorch

Python 3.10/3.11 is recommended.

```bash
python3 -m venv .venv_acme_pytorch
source .venv_acme_pytorch/bin/activate
pip install numpy pandas torch matplotlib "gymnasium[mujoco]" tensorboard wandb "shimmy[bsuite,atari,dm-control]"


python src/train_custom_acme_pytorch_mpo.py --max_steps 300000 --wandb_project acme_pytorch_1 --env_names walker::walk,humanoid::run,cartpole::balance,walker::run,reacher::easy,reacher::hard,hopper::hop,walker::stand,acrobot::swingup,swimmer::swimmer6,swimmer::swimmer15,pendulum::swingup,cheetah::walk,cheetah::run --env_iterations 1
```

### visualize Acme in PyTorch

```bash
python src/visualize_wandb.py --project-metric "acme_pytorch_3::eval/mean_reward"  --entity "adrian-research" --cache-dir logs --show-individual --output results/custom_acme_pytorch.png
```

![Acme PyTorch Results](results/custom_acme_pytorch.png)

## Original MPO Paper

![MPO Original Results](results/original_mpo_paper.png)
