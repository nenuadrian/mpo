#!/usr/bin/env bash

pairs=(
    "cheetah run"
    "hopper stand"
    "walker walk"
    "cartpole balance"
    "walker run"
    "reacher easy"
    "reacher hard"
    "hopper hop"
    "walker stand"
    "acrobot swingup"
    "swimmer swimmer6"
    "swimmer swimmer15"
    "pendulum swingup"
)

for pair in "${pairs[@]}"; do
    read domain task <<< "${pair}"
    
    echo "Running: ${domain}/${task}"
    
    python src/acme_tf_custom/train_acme_tf_mpo.py \
        --domain "${domain}" \
        --task "${task}" \
        --max_actor_steps 10000000 \
        --n_step 5 --timeout 1800 --wandb_project "train_acme_tf_mpo_5"
done
