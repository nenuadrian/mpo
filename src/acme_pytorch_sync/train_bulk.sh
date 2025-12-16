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
    
    python train_custom_acme_pytorch_mpo.py \
        --env_name "${domain}::${task}" \
        --max_actor_steps 3000000 \
        --n_step 5 --wandb_project "acme_pytorch_mpo_sync_no_action_penalization" --action_penalization false
done
