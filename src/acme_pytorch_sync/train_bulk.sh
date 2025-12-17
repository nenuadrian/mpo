#!/usr/bin/env bash

pairs=(
    "cheetah run"
    "hopper stand"
    "walker walk"
    "cartpole balance"
    "walker run"
    "reacher hard"
    "hopper hop"
    "walker stand"
    "acrobot swingup"
    "reacher easy"
    "swimmer swimmer6"
    "swimmer swimmer15"
    "pendulum swingup"
)

for pair in "${pairs[@]}"; do
    read domain task <<< "${pair}"
    
    echo "Running: ${domain}/${task}"
    
    python train_custom_acme_pytorch_mpo.py \
        --env_name "${domain}::${task}" \
        --max_actor_steps 2000000 \
        --n_step 5 --wandb_project "acme_pytorch_mpo_sync_no_AP_DC" --action_penalization false --per_dim_constraining false
done
