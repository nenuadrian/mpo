# PyTorch MPO — Research Pseudocode and Differences from Original MPO Paper

This document provides research-level pseudocode for the PyTorch implementation of MPO included in this repo and a concise checklist of differences compared to the original MPO paper (Abdolmaleki et al., 2018).

Dual/optimization schedule: the paper describes solving the E-step / dual updates and then performing the M-step; here duals, policy and critic are updated together in the same backward pass. This is a pragmatic choice and can work, but is not the strict alternating solver described in some presentations of MPO.

## High-level pseudocode (research level)

Notation:
- B: batch size
- N: number of action samples per batch element
- D: action dimension
- env: environment
- π_θ: parametric policy (Gaussian with mean and scale heads)
- q_φ: critic (Q network)
- π_target, q_target: target policy/critic (hard copies / periodically updated)
- α_mean, α_std: duals (for KL on mean/stddev), T: temperature dual (E-step)
- ε, ε_mean, ε_stddev: MPO constraints
- Replay: off‑policy replay buffer (n-step returns)

Offline/actor-learner loop (actor inserts, learner samples):

1. Actor: collect transitions -> convert to n-step tuples -> insert into Replay.
2. Learner: sample batch of B from Replay (each sample contains obs, action, reward, discount, next_obs).

Compute TD target (using target networks and sampling):
3. For each next state in batch, sample N actions from π_target:
   a. t_mean, t_scale = π_target(next_obs)  # shapes (B,D)
   b. Expand and sample: a_i = t_mean + t_scale * ε_i  for i=1..N  # (N,B,D)
   c. q_i = q_target(next_obs, a_i)  # (N,B)
   d. q_samples = clamp_nan_inf(q_i)
   e. q_t = mean_i[q_samples]  # (B,)  # or other aggregation if desired

Critic (TD) loss:
4. target = reward + discount * q_t
5. q_pred = q_φ(obs, action)  # (B,)
6. critic_loss = 0.5 * mean((target - q_pred)^2)

Policy E-step (compute nonparametric weights):
7. For each batch element b:
   a. For i in 1..N: Q_i = q_samples[i,b]
   b. tempered = detach(Q) / T  # T is softplus(log_T) + eps
   c. w_i = softmax_i(tempered)  # over i, then detach weights
   d. temperature dual loss: L_T = T * (ε + mean_b[logsumexp(tempered)] - log(N) )

(Optionally) MO-MPO action-penalty:
8. Compute action-bound penalty costs and compute penalty weights & penalty temperature loss, then add to weights and temperature loss exactly as E-step.

Policy M-step (fit parametric policy):
9. Define two fixed components (mirroring TF design):
   - fixed_stddev: Normal(mean=π_θ_mean, scale=target_scale)   # keep policy mean variable, use target_scale
   - fixed_mean: Normal(mean=target_mean, scale=π_θ_scale)     # keep policy stddev variable, use target_mean

10. Cross-entropy losses:
   a. L_policy_mean = - mean_b[ sum_i w_i * log_prob_fixed_stddev(a_i) ]
   b. L_policy_stddev = - mean_b[ sum_i w_i * log_prob_fixed_mean(a_i) ]
   c. L_policy = L_policy_mean + L_policy_stddev

Parametric KL penalties & dual updates:
11. Compute parametric KLs per batch:
   a. kl_mean = KL( N(target_mean, target_scale^2) || N(π_θ_mean, target_scale^2) )  # per-dim or aggregated
   b. kl_std = KL( N(target_mean, target_scale^2) || N(target_mean, π_θ_scale^2) )
12. mean_kl = mean_b[kl_*]  # reduce batch mean to get per-dim or scalar
13. loss_kl = sum(detached(alpha) * mean_kl)
14. loss_alpha = sum(alpha * (ε_* - detached(mean_kl)))
15. Combine dual losses: L_dual = loss_alpha_mean + loss_alpha_std + L_T (+ penalty temperature loss if present)

Total policy loss and optimization:
16. L_total = L_policy + loss_kl + L_dual
17. Backprop:
   - Update critic params via critic_loss
   - Update policy params via L_total (policy optimizer)
   - Update dual params via L_total (dual optimizer or separate optimizer)
   - Apply gradient clipping if enabled.
18. Periodically hard-update target networks (policy and critic) per configured intervals.

Diagnostics:
19. Log q_min/q_max, KL diagnostics (relative to ε), dual values, policy stddev statistics, and losses.

Implementation/practical notes:
- Detach semantics matter: q-values used to compute weights are detached; temperature is optimized via its own loss using retained parameter.
- Dual variables are parameterized in log-space and transformed with softplus for positivity; logs are clamped to avoid underflow.
- Numerical stabilizations: softplus scaling, eps additions, torch.nan_to_num applied to q_samples, clamp actions to action bounds.
- Use n-step returns in Replay to better propagate rewards.
- Separate optimizers for policy, critic and dual variables and separate local copies for actor/evaluator threads.

## Where this implementation differs from the original MPO paper

1. Off‑policy with replay and target networks vs. original mostly on‑policy formulation:
   - Original MPO experiments often used on‑policy or fresh-sample sets. This codebase is ACME‑style off‑policy using a replay buffer and target policy/critic networks; this changes stability considerations and requires guard rails (target networks, n-step returns, gradient clipping).

2. Target networks and hard updates:
   - This implementation keeps periodic hard copies of policy and critic (and target obs encoder). The paper uses a more on‑policy sampling regime; implementing stable off‑policy learning required these target copies and different synchronization.

3. Action penalization (MO‑MPO) option:
   - The code includes an optional action out‑of‑bounds penalty and a separate penalty temperature dual (epsilon_penalty). The original MPO paper does not include this exact MO‑MPO penalty variant as implemented.

4. Architecture and normalization differences:
   - The code uses LayerNorm on the observation encoder and ELU activations; the original used different network/initialization choices in some experiments. Final linear layers are initialized near zero to stabilize early learning.

5. Dual management and numerical details:
   - Duals are parameterized as log-variables and softplus-transformed; logs are clamped to min values to avoid numerical issues (implementation detail ensuring stable optimization). Softplus scaling to match initial temperature is included.

6. Per‑dim vs aggregated KL constraints:
   - This implementation supports both per-dimension KL constraints (per_dim_constraining) and aggregate constraints; the original work sometimes reports aggregate constraints — this code exposes both.

7. Sampling, shapes and distribution choices:
   - Uses PyTorch distributions (Independent Normal). The code explicitly constructs "fixed_mean" and "fixed_stddev" components and computes losses accordingly; attention is paid to shapes (N,B,D) and broadcasting to match TF-style behavior.

8. Temperature & weight computation semantics:
   - The E-step uses detached q-values and computes softmax over action samples axis with exact TensorFlow-like formulations (logsumexp and temperature dual loss) to reproduce TF learner diagnostics and behavior.

9. Practical RL engineering choices:
   - Gradient clipping, torch.nan_to_num, explicit dtype/device handling, separate evaluators/actors with local module copies, and using torchrl's TensorDictReplayBuffer for convenience — all practical implementation choices to stabilize off‑policy training not specified in paper.

10. Learning schedule and optimizers:
   - Separate optimizers for critic, policy and duals with different learning rates, and distinct update schedules (e.g., policy and critic periodic target copying) — practical engineering differences from a pure algorithmic description.

References and further reading:
- Abdolmaleki, A., et al. “Maximum a Posteriori Policy Optimisation” (MPO), 2018.
- This README documents implementation choices and numerics needed to make MPO stable in an off‑policy, multi‑threaded PyTorch codebase.
