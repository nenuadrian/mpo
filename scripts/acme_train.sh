#!/usr/bin/env bash

pairs=(
#  "hopper stand"
#  "walker walk"
  "humanoid run"
  "cartpole balance"
  "walker run"
  "reacher easy"
  "reacher hard"
  "hopper hop"
  "walker stand"
  "acrobot swingup"
  "swimmer swimmer6"
  "sweimmer swimmer15"
  "pendulum swingup"
  "cheetah walk"
  "cheetah run"
)

mkdir logs

for pair in "${pairs[@]}"; do
  read domain task <<< "${pair}"

  echo "Running: ${domain}/${task}"

  python src/train_acme_lp_tf_mpo.py \
    --domain "${domain}" \
    --task "${task}" \
    --max_actor_steps 10000000 > logs/logs-$domain-$task.log 2>&1
done