import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    MLP Q-network expecting inputs (states, actions).
    Provides:
      - forward(states, actions) -> [B] tensor of Q-values
      - retrace_targets(...) helper to compute Retrace-style targets
    Note: retrace_targets implements a practical, sample-based Retrace-ish
    estimator suitable for use with an n-step replay buffer where only single-step
    behavior log-probs are available. It approximates E_{a'~pi}[Q(s',a')] by
    sampling multiple actions and applies a truncated importance weight
    c = min(1, pi_log_prob - b_log_prob).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        input_dim = obs_dim + act_dim
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # states: [B, obs_dim], actions: [B, act_dim] -> returns [B]
        x = torch.cat([states, actions], dim=-1)
        out = self.net(x).squeeze(-1)
        return out

    @torch.no_grad()
    def retrace_targets(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
        logp_mu: torch.Tensor,
        pi: nn.Module,
        q_target: "QNetwork",
        gamma: float = 0.99,
        num_action_samples: int = 8,
    ):
        """
        Compute Retrace-style targets for a batch.
        Args:
          states, actions, rewards, dones, next_states: tensors shaped [B,...]
          logp_mu: tensor with behavior log-prob for sampled actions [B] (may be zeros)
          pi: current policy (must implement .sample and .log_prob)
          q_target: target Q-network (callable like forward)
          gamma: discount
          num_action_samples: number of actions to sample per next-state for E_pi[Q]
        Returns:
          target_q: tensor [B] for use with MSE loss against q(states,actions)
        Notes:
          - If logp_mu contains zeros (or is not meaningful), importance weights fallback to 1 (clamped).
          - This is a practical sample-based Retrace approximation (one-step corrected).
        """
        device = states.device
        B = states.shape[0]

        # Estimate E_{a'~pi}[ Q_target(next_state, a') ] by sampling multiple actions
        if num_action_samples <= 1:
            # single sample (fast path)
            next_actions, _ = pi.sample(next_states)
            q_next = q_target(next_states, next_actions)
        else:
            # Expand next_states to [B * M, obs_dim]
            next_states_exp = next_states.unsqueeze(1).expand(
                -1, num_action_samples, -1
            )
            next_states_flat = next_states_exp.reshape(B * num_action_samples, -1)
            actions_flat, _ = pi.sample(next_states_flat)
            # Evaluate Q on flattened batch and average per-state
            q_next_flat = q_target(next_states_flat, actions_flat)  # [B*M]
            q_next = q_next_flat.view(B, num_action_samples).mean(dim=1)

        # standard one-step target (expected under pi)
        target = rewards + gamma * (1.0 - dones) * q_next

        # compute truncated importance weight for the sampled actions: c = min(1, exp(log_pi - logp_mu))
        try:
            log_pi_sample = pi.log_prob(states, actions)  # [B]
        except Exception:
            # if policy doesn't support log_prob signature used elsewhere, try flipped args
            log_pi_sample = pi.log_prob(actions, states)

        # Ensure tensors are same dtype/device
        logp_mu_t = (
            logp_mu.to(device=device, dtype=log_pi_sample.dtype)
            if isinstance(logp_mu, torch.Tensor)
            else torch.tensor(logp_mu, device=device, dtype=log_pi_sample.dtype)
        )

        # numeric stability: where behavior log-prob is zero (or extremely small), fall back to weight 1.
        # compute ratio = exp(log_pi - logp_mu) then trunc to 1
        ratio = torch.exp(log_pi_sample - logp_mu_t)
        c = torch.minimum(ratio, torch.ones_like(ratio))

        # Final retrace-style target (one-step truncated correction):
        # target_retrace = q(s,a) + c * (target - q(s,a))
        # Return full target; caller typically computes loss against q(states,actions)
        return target * c + (1.0 - c) * self(states, actions)
