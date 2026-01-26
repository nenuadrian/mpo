import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class DisentangledTransformerMD(nn.Module):
    """
    Three-layer disentangled Transformer implementing one-step Mirror Descent for an MTD model.
    This model uses only attention (no MLPs) and relative positional encodings as described in Proposition 3.
    """

    def __init__(
        self,
        vocab_size: int,
        m: int,
        transition_matrix: torch.Tensor,
        beta: float = 1.0,
    ):
        """
        Initialize the Transformer.
        Args:
            vocab_size (int): Vocabulary size (q).
            m (int): MTD model order (number of past lags to consider).
            transition_matrix (torch.Tensor): Base transition matrix π* of shape (q, q).
                                              π[i, j] = P(next_token=j | context_token=i).
            beta (float): Scaling factor (learning rate) for the Mirror Descent update.
        """
        super().__init__()
        self.q = vocab_size
        self.m = m
        # Store π* as a constant (no training). Ensure it's a probability matrix (each row sums to 1).
        self.pi = transition_matrix.clone().detach()
        # We'll use log_pi for attention score computations (log probabilities for stability).
        self.log_pi = torch.log(self.pi + 1e-12)
        # Scaling parameter for MD update (corresponds to β in the paper).
        self.beta = beta

    def forward(self, sequence: list) -> torch.Tensor:
        """
        Forward pass: Given a token sequence (list of indices), return the next-token predictive distribution.
        The output is a tensor of length q with probabilities for each possible next token.
        """
        # Convert sequence to tensor of token indices
        seq = torch.tensor(sequence, dtype=torch.long)
        T = seq.size(0)
        if T < 1:
            raise ValueError("Sequence must contain at least 1 token.")
        # If sequence length <= m, we have no in-context evidence to update λ (use prior uniform mixture).
        if T <= self.m:
            # Predict with uniform mixture weights (1/m for each lag)
            # Distribution = (1/m) * sum_{g=1..m} π(y_{T+1-g}, :)
            # (For T < m, some terms skip because context not available beyond start)
            pred_probs = torch.zeros(self.q)
            for g in range(1, self.m + 1):
                if g > T:
                    break
                ctx_token = seq[T - g].item()
                pred_probs += (1.0 / self.m) * self.pi[ctx_token]
            return pred_probs

        # Layer 1: Compute posterior responsibilities γ_i(g) for each position i and lag g.
        # γ_i(g) = P(Z_i = g | Y_{1:i}, λ prior uniform)
        #        = π(Y_{i-g}, Y_i) / sum_{h=1..avail} π(Y_{i-h}, Y_i),
        # where 'avail' = min(m, i-1) for i <= m, or = m for i > m.
        T_idx = T  # number of tokens
        gamma = torch.zeros((T_idx, self.m))
        # Note: We use 1-indexed notation in comments (positions 1..T), but code is 0-indexed.
        for i in range(
            1, T_idx
        ):  # start from second token (i_idx=1 corresponds to position 2)
            # Determine available lags for position i+1 (1-index). If i_idx = i (0-index), that's position i+1.
            avail_lags = min(self.m, i)
            # Current token at position i+1 (1-index) is seq[i] (0-index)
            current_token = seq[i].item()
            # Compute unnormalized probabilities for each lag g <= avail_lags
            # Using log_pi for numerical stability: log_weight_g = log λ_g (prior uniform => log(1) = 0) + log π(Y_{i+1-g}, Y_{i+1})
            # Since prior is uniform, effectively weight ∝ π(Y_{i-g}, Y_i).
            # We'll compute in probability space here (since m is small typically).
            total_prob = 0.0
            probs = []
            for g in range(1, avail_lags + 1):
                prev_token = seq[i - g].item()  # token at position (i+1-g)
                p = self.pi[prev_token, current_token].item()
                probs.append(p)
                total_prob += p
            if total_prob <= 0:
                continue  # if no probability mass (should not happen if π is valid), skip (γ remains all-zero)
            # Normalize to get responsibilities
            for g_idx, p in enumerate(probs, start=1):
                gamma[i, g_idx - 1] = p / total_prob
            # Note: for g > avail_lags, gamma remains 0 (no influence from those lags if i < g).

        # Layer 2: Aggregate (average) the responsibility vectors over positions i > m (i = m+1..T in 1-index).
        # Compute the average γ vector from positions m (0-index) to T-1 (inclusive).
        # (These correspond to positions m+1..T in 1-index, the portion of the sequence with full context.)
        # We average rather than sum to align with scaling in the Mirror Descent update (Equation 7 includes 1/(T-m) factor).
        gamma_sum = gamma[self.m : T_idx].sum(dim=0)  # sum over rows m..T-1
        count = T_idx - self.m  # number of terms summed (T - m)
        gamma_avg = gamma_sum / count  # average responsibility for each lag g

        # Layer 3: Compute the one-step mixture weight estimate λ~ via a softmax of the scaled average γ.
        # λ~_g = softmax( β * γ_avg )_g
        # This corresponds to one step of Mirror Descent starting from uniform weights.
        lambda_est = F.softmax(self.beta * gamma_avg, dim=0)  # shape: (m,)

        # Output linear layer: Map the mixture of previous tokens (weighted by λ~) to predictive logits via π*.
        # Predictive distribution: P(Y_{T+1}=x | context) = sum_{g=1}^m λ~_g * π(Y_{T+1-g}, x)
        pred_probs = torch.zeros(self.q)
        # Use the last m tokens in the sequence as context (positions T-m+1 .. T in 1-index).
        for g in range(1, self.m + 1):
            if g > T_idx:
                break  # (safety check; T_idx >= m here so this should not trigger)
            ctx_token = seq[T_idx - g].item()  # token at position T+1-g (1-index)
            pred_probs += lambda_est[g - 1] * self.pi[ctx_token]
        return pred_probs


# Utility Functions


def mirror_descent_estimator(
    sequence: list, pi_matrix: torch.Tensor, m: int, beta: float = 1.0
):
    """
    Compute the one-step Mirror Descent mixture weight estimate and predictive distribution directly (analytic solution).
    Returns a tuple (lambda_est, pred_dist):
      - lambda_est: np.ndarray of shape (m,) for the estimated mixture weights λ~.
      - pred_dist: np.ndarray of shape (q,) for the next-token predictive probability distribution.
    """
    T = len(sequence)
    q = pi_matrix.size(0)
    # If no evidence beyond initial context (T <= m), mixture remains uniform.
    if T <= m:
        lambda_est = np.ones(m) / m
        # Predict with uniform λ: distribution = 1/m * sum_{g=1..m} π(y_{T+1-g}, :)
        pred_probs = np.zeros(q)
        for g in range(1, m + 1):
            if g > T:
                break
            ctx_token = sequence[T - g]
            pred_probs += (1.0 / m) * pi_matrix[ctx_token].numpy()
        return lambda_est, pred_probs

    # Compute responsibilities γ_i(g) for i = 1..T (1-index), using uniform prior.
    gamma = np.zeros((T, m))
    for i_idx in range(1, T):
        avail_lags = min(m, i_idx)
        curr_token = sequence[i_idx]
        # Compute unnormalized probabilities for each g <= avail_lags
        contrib = []
        for g in range(1, avail_lags + 1):
            prev_token = sequence[i_idx - g]
            contrib.append(pi_matrix[prev_token, curr_token].item())
        total = sum(contrib)
        if total <= 0:
            continue
        # Normalize contributions to get gamma[i_idx, g-1]
        for g_idx, p in enumerate(contrib, start=1):
            gamma[i_idx, g_idx - 1] = p / total

    # Average γ over positions m..T-1 (0-index), i.e. positions m+1..T in 1-index.
    gamma_avg = gamma[m:T].mean(axis=0)  # average along axis 0
    # Compute λ~ via softmax(β * gamma_avg)
    x = beta * gamma_avg
    # Subtract max for numerical stability
    x = x - np.max(x)
    exp_x = np.exp(x)
    lambda_est = exp_x / exp_x.sum()
    # Compute predictive distribution: sum_{g=1..m} λ~_g * π(y_{T+1-g}, :)
    pred_probs = np.zeros(q)
    for g in range(1, m + 1):
        if g > T:
            break
        ctx_token = sequence[T - g]
        pred_probs += lambda_est[g - 1] * pi_matrix[ctx_token].numpy()
    return lambda_est, pred_probs


def bayes_optimal_estimator(
    sequence: list,
    pi_matrix: torch.Tensor,
    m: int,
    num_samples: int = 10000,
    burnin: int = 5000,
):
    """
    Approximate the Bayes-optimal mixture and predictive distribution via MCMC.
    Uses Gibbs sampling to draw posterior samples of λ given the sequence.
    Returns (lambda_mean, pred_dist):
      - lambda_mean: np.ndarray of shape (m,) for the posterior mean of λ.
      - pred_dist: np.ndarray of shape (q,) for the Bayes-predictive distribution of the next token.
    """
    T = len(sequence)
    q = pi_matrix.size(0)
    if T <= m:
        # With no evidence beyond context, posterior of λ remains Dirichlet(1,...,1) (uniform).
        lambda_mean = np.ones(m) / m
        pred_probs = np.zeros(q)
        for g in range(1, m + 1):
            if g > T:
                break
            ctx_token = sequence[T - g]
            pred_probs += (1.0 / m) * pi_matrix[ctx_token].numpy()
        return lambda_mean, pred_probs

    # Gibbs sampling of λ and latent assignments Z_{m+1}..Z_T.
    # Initialize λ (start from prior Dirichlet(1) sample or uniform).
    lambda_curr = np.random.dirichlet(np.ones(m))
    lambda_sum = np.zeros(m)
    samples_collected = 0
    seq_arr = np.array(sequence, dtype=int)
    for it in range(num_samples):
        # Sample latent lag assignments Z_{t} for t = m+1..T
        Z_samples = []
        for idx in range(m, T):
            # idx (0-index) corresponds to time t = idx+1 (1-index)
            curr_token = seq_arr[idx]  # Y_t
            # Compute weight for each lag g
            weights = []
            for g in range(1, m + 1):
                if g > idx:  # not enough context (shouldn't happen for idx >= m)
                    weights.append(0.0)
                else:
                    prev_token = seq_arr[idx - g]  # Y_{t-g}
                    # weight ∝ λ_g * π(Y_{t-g}, Y_t)
                    weights.append(
                        lambda_curr[g - 1] * pi_matrix[prev_token, curr_token].item()
                    )
            # Normalize weights to probabilities
            total = sum(weights)
            if total <= 0:
                weights = [
                    1.0 / m
                ] * m  # if no probability (degenerate case), assume uniform
            else:
                weights = [w / total for w in weights]
            # Sample Z_t = g according to these probabilities
            g_choice = random.choices(range(1, m + 1), weights=weights, k=1)[0]
            Z_samples.append(g_choice)
        # Given sampled Z, update λ by sampling from Dirichlet(posterior)
        counts = [0] * m
        for g in Z_samples:
            counts[g - 1] += 1
        # Posterior α = prior(=1) + counts
        alpha_post = np.array([1.0 + c for c in counts])
        lambda_curr = np.random.dirichlet(alpha_post)
        # After burn-in, accumulate λ samples for expectation
        if it >= burnin:
            lambda_sum += lambda_curr
            samples_collected += 1

    # Posterior mean of λ
    lambda_mean = (
        lambda_sum / samples_collected if samples_collected > 0 else lambda_curr
    )
    # Compute Bayes-predictive distribution: E[ P(Y_{T+1}|λ) ] = P(Y_{T+1}| E[λ|data] )
    # (Linear in λ, so we use mean λ for predictive distribution.)
    pred_probs = np.zeros(q)
    for g in range(1, m + 1):
        if g > T:
            break
        ctx_token = seq_arr[T - g]
        pred_probs += lambda_mean[g - 1] * pi_matrix[ctx_token].numpy()
    return lambda_mean, pred_probs


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL divergence KL(P || Q) between two discrete distributions p and q.
    Both p and q are 1D arrays that sum to 1. Returns KL(p||q).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    eps = 1e-12
    # Avoid division by zero by adding a tiny epsilon to q and renormalizing
    q = q + eps
    q = q / np.sum(q)
    mask = p > 1e-12  # consider terms where p_i > 0
    p_masked = p[mask]
    q_masked = q[mask]
    # Compute sum_i p_i * log(p_i / q_i)
    return float(np.sum(p_masked * np.log(p_masked / q_masked)))
