"""Replay buffer for storing sequences for Retrace-style off-policy learning.

This module implements a circular replay buffer that stores fixed-length
sequences of transitions. This is suitable for algorithms like Retrace
that require a trajectory segment to compute targets.
"""

from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch


class NStepReplayBuffer:
    """n-step sequence circular replay buffer.

    Parameters
    - obs_shape (tuple): shape of a single observation.
    - act_shape (tuple): shape of a single action.
    - capacity (int): maximum number of sequences to keep.
    - n_step (int): sequence length.
    - gamma (float): discount factor (not used in buffer, but kept for API).
    - device (str or torch.device): device for sampled tensors.

    Behavior notes
    - The buffer stores sequences of transitions of length up to `n_step`.
    - If an episode ends, a partial sequence is stored. The actual length
      of each stored sequence is also recorded.
    """

    capacity: int
    n_step: int
    gamma: float
    device: Union[str, torch.device]

    obs: np.ndarray
    next_obs: np.ndarray
    acts: np.ndarray
    rews: np.ndarray
    dones: np.ndarray
    logp_mu: np.ndarray
    seq_lens: np.ndarray

    idx: int
    full: bool
    n_step_buffer: List[Dict[str, Any]]

    def __init__(
        self,
        obs_shape: Sequence[int],
        act_shape: Sequence[int],
        capacity: int,
        n_step: int = 1,
        gamma: float = 0.99,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.device = device

        self.obs = np.zeros((capacity, n_step) + tuple(obs_shape), dtype=np.float32)
        self.acts = np.zeros((capacity, n_step) + tuple(act_shape), dtype=np.float32)
        self.rews = np.zeros((capacity, n_step), dtype=np.float32)
        self.dones = np.zeros((capacity, n_step), dtype=np.bool_)
        self.logp_mu = np.zeros((capacity, n_step), dtype=np.float32)
        self.next_obs = np.zeros((capacity,) + tuple(obs_shape), dtype=np.float32)
        self.seq_lens = np.zeros((capacity,), dtype=np.int32)

        self.idx = 0
        self.full = False
        self.n_step_buffer = []

    def _store_sequence(self, sequence: List[Dict[str, Any]]) -> None:
        """Stores a sequence of transitions."""
        seq_len = len(sequence)

        obs_seq = np.array([t["obs"] for t in sequence])
        acts_seq = np.array([t["act"] for t in sequence])
        rews_seq = np.array([t["rew"] for t in sequence])
        dones_seq = np.array([t["done"] for t in sequence])
        logp_mu_seq = np.array([t["logp_mu"] for t in sequence])

        self.obs[self.idx, :seq_len] = obs_seq
        self.acts[self.idx, :seq_len] = acts_seq
        self.rews[self.idx, :seq_len] = rews_seq
        self.dones[self.idx, :seq_len] = dones_seq
        self.logp_mu[self.idx, :seq_len] = logp_mu_seq

        self.next_obs[self.idx] = sequence[-1]["next_obs"]
        self.seq_lens[self.idx] = seq_len

        # Zero out remaining part of the sequence if it's a partial one
        if seq_len < self.n_step:
            self.obs[self.idx, seq_len:] = 0
            self.acts[self.idx, seq_len:] = 0
            self.rews[self.idx, seq_len:] = 0
            self.dones[self.idx, seq_len:] = True
            self.logp_mu[self.idx, seq_len:] = 0

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def push(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        logp_mu: float,
        next_obs: np.ndarray,
    ) -> None:
        """Push a new one-step transition."""
        transition = {
            "obs": obs,
            "act": act,
            "rew": rew,
            "done": done,
            "logp_mu": logp_mu,
            "next_obs": next_obs,
        }
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) >= self.n_step:
            self._store_sequence(self.n_step_buffer)
            self.n_step_buffer.pop(0)

        if done:
            if self.n_step_buffer:
                self._store_sequence(self.n_step_buffer)
            self.n_step_buffer.clear()

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def sample(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample a minibatch of sequences and return torch tensors."""
        assert len(self) >= batch_size
        idxs = np.random.randint(0, len(self), size=batch_size)

        obs = torch.as_tensor(self.obs[idxs], device=self.device)
        acts = torch.as_tensor(self.acts[idxs], device=self.device)
        rews = torch.as_tensor(self.rews[idxs], device=self.device)
        dones = torch.as_tensor(
            self.dones[idxs].astype(np.float32), device=self.device
        )
        next_obs = torch.as_tensor(self.next_obs[idxs], device=self.device)
        logp_mu = torch.as_tensor(self.logp_mu[idxs], device=self.device)
        seq_lens = torch.as_tensor(self.seq_lens[idxs], device=self.device)

        return obs, acts, rews, dones, next_obs, logp_mu, seq_lens

