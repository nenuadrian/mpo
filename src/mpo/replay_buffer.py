"""Simple n-step replay buffer utilities used by MPO experiments.

This module implements a compact, numpy-backed circular replay buffer that
supports n-step returns for bootstrapping. The buffer stores observations,
actions, scalar rewards, done flags and an optional behaviour log-prob
(`logp_mu`) per time step. Samples are returned as PyTorch tensors on a
configured device.

Key concepts and design choices:
- Memory layout: contiguous numpy arrays for each field. This is efficient
  for slicing and converting to torch tensors when sampling.
- Circular buffer: `idx` advances modulo `capacity`; when it wraps to zero
  `full` is set to True and the buffer is considered full. `len(buffer)` is
  implemented to return the current number of stored elements.
- n-step returns: a small, temporary `n_step_buffer` accumulates incoming
  one-step transitions; once it contains `n_step` elements a single
  n-step transition is emitted to the main storage. If an episode terminates
  before `n_step` is reached the partial buffer is flushed.

This implementation is intentionally small and dependency-free (aside from
numpy and torch). It is suitable for synchronous training loops where
concurrent writes/reads across threads/processes are not performed. If you
need multi-process safety or prioritized replay, consider using more
sophisticated data structures.
"""

from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch


class NStepReplayBuffer:
    """n-step circular replay buffer.

    Parameters
    - obs_shape (tuple): shape of a single observation (e.g., (3, 84, 84)).
    - act_shape (tuple): shape of a single action (e.g., () for scalar, or
      (n,) for vector actions). The implementation stores actions as floats
      (dtype=np.float32); if actions are discrete, they will be stored as
      float values and converted back by the user if needed.
    - capacity (int): maximum number of (n-step) transitions to keep. When
      capacity is reached the buffer overwrites oldest elements (circular).
    - n_step (int): number of steps to accumulate for n-step returns.
    - gamma (float): discount factor used when computing n-step returns.
    - device (str or torch.device): device onto which sampled tensors are
      moved (e.g., 'cpu' or 'cuda').

    Behavior notes
    - The buffer stores the n-step return R = sum_{t=0}^{n-1} gamma^t r_t and
      the `next_obs` is the observation after the last step in the n-step
      sequence. The `done` flag is taken from the last step; if any earlier
      step had `done=True` the accumulation stops at that point.
    - `logp_mu` is intended to store the behaviour policy log-probability of
      the stored action and is optional in the sense that algorithms may set
      it to 0 if unused.
    """

    # Class attribute type annotations for static checkers and IDEs.
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
        # Basic configuration
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.device = device

        # Main storage arrays. Each index i corresponds to one (possibly
        # n-step) transition: obs[i], acts[i], rews[i], dones[i], next_obs[i].
        # Using numpy arrays lets us efficiently slice with an index array
        # during sampling, then convert the whole batch to torch tensors.
        self.obs = np.zeros((capacity,) + tuple(obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity,) + tuple(obs_shape), dtype=np.float32)
        self.acts = np.zeros((capacity,) + tuple(act_shape), dtype=np.float32)
        # Rewards and done flags are stored as 1D arrays for compactness.
        self.rews = np.zeros((capacity,), dtype=np.float32)
        # Use numpy bool_ for done flags; converted to float32 on sample.
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        # Optional behaviour log probability of the stored action. This is
        # commonly used by importance-sampling / off-policy algorithms.
        self.logp_mu = np.zeros((capacity,), dtype=np.float32)

        # Circular buffer bookkeeping: `idx` points to the next write index.
        # `full` toggles when we've wrapped around at least once.
        self.idx = 0
        self.full = False

        # Temporary buffer used to accumulate raw (one-step) transitions for
        # n-step bootstrapping. Each element is a small dict containing the
        # fields of a single step. This buffer is cleared at episode end.
        self.n_step_buffer = []

    def _store_transition(
        self,
        o: np.ndarray,
        a: np.ndarray,
        r: float,
        done: bool,
        logp_mu: float,
        o_next: np.ndarray,
    ) -> None:
        """Write a single (already-constructed) transition into main storage.

        This function handles the circular overwrite logic. It assumes the
        provided values are already the intended n-step aggregated values
        (i.e., `r` may be an n-step return and `o_next` the observation after
        n steps).

        Parameters
        - o: observation at time t (shape matches `obs_shape`).
        - a: action taken at time t (shape matches `act_shape`).
        - r: scalar reward (possibly n-step aggregated).
        - done: boolean indicating whether the episode ended within the
          n-step window (taken from the last step in that window).
        - logp_mu: behaviour policy log-probability for the action.
        - o_next: observation following the last step in the n-step window.
        """
        # Store values into the arrays at the current write index.
        self.obs[self.idx] = o
        self.acts[self.idx] = a
        self.rews[self.idx] = r
        self.dones[self.idx] = done
        self.next_obs[self.idx] = o_next
        self.logp_mu[self.idx] = logp_mu

        # Advance write pointer with wrap-around. When we wrap to zero we
        # mark the buffer as full so that __len__ returns capacity.
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
        """Push a new one-step transition and possibly emit an n-step entry.

        The method operates in two phases:
        1. Append the incoming one-step transition to `n_step_buffer`.
        2. If `n_step_buffer` contains at least `n_step` elements, compute the
           n-step return R = sum_{t=0}^{n-1} gamma^t r_t (stopping early if a
           terminal is encountered), then emit a consolidated transition to
           main storage using the first observation/action and the last
           observation as `next_obs`.

        When a terminal `done=True` is observed, any partial contents of the
        `n_step_buffer` are cleared because they cannot be completed across
        episode boundaries.
        """
        transition = {
            "obs": obs,
            "act": act,
            "rew": rew,
            "done": done,
            "logp_mu": logp_mu,
            "next_obs": next_obs,
        }
        self.n_step_buffer.append(transition)

        # If we have enough steps to form an n-step return, compute it and
        # store an aggregated transition into the main circular buffer.
        if len(self.n_step_buffer) >= self.n_step:
            R, gamma_pow = 0.0, 1.0
            # Accumulate discounted rewards up to n steps or until done.
            for t in range(self.n_step):
                tr = self.n_step_buffer[t]
                R += gamma_pow * tr["rew"]
                gamma_pow *= self.gamma
                # If any step is terminal, stop accumulating further future
                # rewards (standard n-step handling).
                if tr["done"]:
                    break

            # The emitted transition uses the first step's observation/action
            # and the last step's `next_obs` and done flag.
            first = self.n_step_buffer[0]
            last = self.n_step_buffer[self.n_step - 1]

            self._store_transition(
                o=first["obs"],
                a=first["act"],
                r=R,
                done=last["done"],
                logp_mu=first["logp_mu"],
                o_next=last["next_obs"],
            )

            # Slide the buffer forward by removing the earliest step. This
            # implements a sliding window so overlapping n-step transitions
            # are available for subsequent pushes.
            self.n_step_buffer.pop(0)

        # If the incoming step ended an episode, clear any remaining partial
        # information in the n-step accumulator. These incomplete sequences
        # cannot be extended across episodes.
        if done:
            self.n_step_buffer.clear()

    def __len__(self) -> int:
        """Return the current number of stored transitions.

        If the circular buffer has wrapped at least once, the buffer is
        considered full and its length is `capacity`. Otherwise the length is
        the current write pointer `idx` which indicates how many items have
        been written so far.
        """
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample a minibatch of transitions and return torch tensors.

        The function randomly samples `batch_size` indices uniformly from the
        range [0, len(buffer)). It converts the selected numpy slices into
        PyTorch tensors and moves them to `self.device`.

        Returned tensors (in order):
        - obs: float32 tensor of shape (B, *obs_shape)
        - acts: float32 tensor of shape (B, *act_shape)
        - rews: float32 tensor of shape (B,)
        - dones: float32 tensor of shape (B,) where True->1.0 and False->0.0
        - next_obs: float32 tensor of shape (B, *obs_shape)
        - logp_mu: float32 tensor of shape (B,)

        Note: this sampling is uniform and independent across indices; no
        sequence or temporal adjacency is enforced by `sample` itself.
        """
        assert len(self) >= batch_size
        # Choose random indices from the currently filled portion of the
        # circular buffer. `len(self)` respects whether the buffer has
        # wrapped.
        idxs = np.random.randint(0, len(self), size=batch_size)

        # Convert numpy batches to torch tensors on the configured device.
        obs = torch.as_tensor(self.obs[idxs], device=self.device)
        acts = torch.as_tensor(self.acts[idxs], device=self.device)
        rews = torch.as_tensor(self.rews[idxs], device=self.device)
        # Convert boolean done flags to float (0.0/1.0) for numeric ops.
        dones = torch.as_tensor(self.dones[idxs].astype(np.float32), device=self.device)
        next_obs = torch.as_tensor(self.next_obs[idxs], device=self.device)
        logp_mu = torch.as_tensor(self.logp_mu[idxs], device=self.device)

        return obs, acts, rews, dones, next_obs, logp_mu
