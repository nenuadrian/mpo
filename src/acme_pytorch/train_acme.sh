#!/usr/bin/env bash

pairs=(
  "hopper stand"
  "hopper hop"
  "cartpole balance"
  "walker walk"
  "walker run"
  "walker stand"
  "reacher easy"
  "reacher hard"
  "acrobot swingup"
  "swimmer swimmer6"
  "swimmer swimmer15"
  "pendulum swingup"
  "cheetah run"
)

mkdir logs

for pair in "${pairs[@]}"; do
  read domain task <<< "${pair}"

  echo "Running: ${domain}/${task}"

  python train.py \
    --domain "${domain}" \
    --task "${task}" \
    --max_actor_steps 10000000 > logs/logs-$domain-$task.log 2>&1
done