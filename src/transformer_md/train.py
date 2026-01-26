"""
Transformers Learn Latent Mixture Models In-Context via Mirror Descent
- Constructed 3-layer *disentangled* transformer implementing one-step MD for MTD.
- Synthetic MTD sequence generator.
- Direct one-step MD estimator for comparison.
- Standard Transformer skeleton (QKV) for comparison (not trained).

Tested with PyTorch >= 2.1

Paper: "TRANSFORMERS LEARN LATENT MIXTURE MODELS IN-CONTEXT VIA MIRROR DESCENT" (ICLR 2026 submission)
(see provided PDF)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def row_stochastic(mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize rows to sum to 1."""
    mat = mat.clamp_min(eps)
    return mat / mat.sum(dim=-1, keepdim=True).clamp_min(eps)


def causal_mask(T: int, device=None) -> torch.Tensor:
    """Mask with True where j <= i (allowed)."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return j <= i


# -----------------------------
# MTD synthetic data generator
# -----------------------------


@dataclass
class MTDConfig:
    q: int  # alphabet size
    m: int  # order / max lag
    T: int  # sequence length


def sample_pi_star(q: int, concentration: float = 1.0, device=None) -> torch.Tensor:
    """
    Sample a random transition matrix pi_star in Δ^{q-1} row-wise:
      pi_star[i, :] ~ Dirichlet(concentration * 1_q)
    """
    alpha = torch.full((q,), float(concentration), device=device)
    dist = torch.distributions.Dirichlet(alpha)
    rows = dist.sample((q,))  # [q, q]
    return row_stochastic(rows)


def sample_lambda_dirichlet(m: int, device=None) -> torch.Tensor:
    """λ ~ Dirichlet(1,...,1)."""
    alpha = torch.ones(m, device=device)
    return torch.distributions.Dirichlet(alpha).sample()  # [m]


@torch.no_grad()
def generate_mtd_sequence(
    cfg: MTDConfig,
    pi_star: torch.Tensor,
    lam: Optional[torch.Tensor] = None,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate one sequence y_1:T under MTD(m):
      For t > m:
        sample Z_t ~ Cat(λ) over {1..m}
        sample Y_t ~ Cat( pi_star[ Y_{t-Z_t}, : ] )
    For t <= m: sample Y_t ~ Uniform over {0..q-1}.
    Returns:
      y: [T] int64 tokens in {0..q-1}
      lam: [m] mixture weights used
    """
    q, m, T = cfg.q, cfg.m, cfg.T
    device = device or pi_star.device
    if lam is None:
        lam = sample_lambda_dirichlet(m, device=device)

    y = torch.empty(T, dtype=torch.long, device=device)
    # init first m tokens uniformly
    y[:m] = torch.randint(low=0, high=q, size=(m,), device=device)

    cat_lam = torch.distributions.Categorical(probs=lam)
    for t in range(m, T):
        # sample lag g in {1..m}
        g = int(cat_lam.sample().item()) + 1
        prev = int(y[t - g].item())
        cat_next = torch.distributions.Categorical(probs=pi_star[prev])
        y[t] = cat_next.sample()

    return y, lam


@torch.no_grad()
def generate_batch(
    cfg: MTDConfig,
    pi_star: torch.Tensor,
    B: int,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return:
    ys: [B, T]
    lams: [B, m]
    """
    device = device or pi_star.device
    ys = torch.empty(B, cfg.T, dtype=torch.long, device=device)
    lams = torch.empty(B, cfg.m, dtype=torch.float, device=device)
    for b in range(B):
        y, lam = generate_mtd_sequence(cfg, pi_star, lam=None, device=device)
        ys[b] = y
        lams[b] = lam
    return ys, lams


# -----------------------------
# Direct one-step MD estimator (paper Eq. 6 / Prop 2)
# -----------------------------


@torch.no_grad()
def mtd_responsibilities_gamma(
    y: torch.Tensor,  # [T] tokens
    pi_star: torch.Tensor,  # [q, q]
    m: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute γ_k(g) for k = m..T-1 (0-indexed time),
    where (paper indexing k = m+1..T):
      γ_k(g) = π(y_{k-g}, y_k) / Σ_{h=1..m} π(y_{k-h}, y_k)

    Returns:
      gamma: [T, m] with gamma[k, g-1] meaningful for k>=m, else 0.
    """
    T = y.shape[0]
    q = pi_star.shape[0]
    assert pi_star.shape == (q, q)
    gamma = torch.zeros(T, m, device=y.device, dtype=pi_star.dtype)

    for k in range(m, T):
        yk = int(y[k].item())
        nums = []
        for g in range(1, m + 1):
            y_prev = int(y[k - g].item())
            nums.append(pi_star[y_prev, yk])
        nums_t = torch.stack(nums, dim=0)  # [m]
        denom = nums_t.sum().clamp_min(eps)
        gamma[k] = nums_t / denom

    return gamma  # [T, m]


@torch.no_grad()
def one_step_md_lambda_hat(
    y: torch.Tensor,  # [T]
    pi_star: torch.Tensor,  # [q, q]
    m: int,
    eta_or_beta: float = 1.0,
    average: bool = True,
) -> torch.Tensor:
    """
    One-step exponentiated-gradient / MD estimator (paper Prop 2 / Eq 6):
      λ̂_g ∝ exp( η * m * Σ_{k=m+1..T} γ_k(g) )
    The constructed transformer uses a scaled form like:
      λ̃_g ∝ exp( β/(T-m) * Σ_{k=m+1..T} γ_k(g) )
    (paper Prop 3 / Eq 7).

    We support both via:
      if average=True: use (eta_or_beta)/(T-m) * Σ gamma
      else: use (eta_or_beta)*m * Σ gamma  (as in Eq 6)

    Returns: [m] on simplex.
    """
    T = y.shape[0]
    gamma = mtd_responsibilities_gamma(y, pi_star, m)  # [T, m]
    s = gamma[m:].sum(dim=0)  # sum over k=m..T-1 corresponds to k=m+1..T in paper
    if average:
        scale = float(eta_or_beta) / max(1, (T - m))
    else:
        scale = float(eta_or_beta) * float(m)
    logits = scale * s
    return F.softmax(logits, dim=-1)  # [m]


@torch.no_grad()
def mtd_predict_next_from_lambda(
    y: torch.Tensor,  # [T]
    pi_star: torch.Tensor,  # [q, q]
    lam: torch.Tensor,  # [m]
    m: int,
) -> torch.Tensor:
    """
    Predict distribution for Y_{T+1}:
      p(next=j) = Σ_{g=1..m} λ_g * π( y_{T+1-g}, j )
    where y_{T+1-g} for g in 1..m refers to y[T-g] (0-indexed).
    Returns: [q]
    """
    T = y.shape[0]
    q = pi_star.shape[0]
    out = torch.zeros(q, device=y.device, dtype=pi_star.dtype)
    for g in range(1, m + 1):
        prev = int(y[T - g].item())
        out = out + lam[g - 1] * pi_star[prev]
    return out


# -----------------------------
# Disentangled Transformer components (paper Sec 2)
# -----------------------------


class DisentangledSingleHead(nn.Module):
    """
    One attention head with:
      e_ij = h_i^T W_A h_j + h_i^T rA_{k} ,  k = (i-j)+1
    causal softmax over j<=i
    output:
      \hat{h}_i = Σ_j A_ij * Concat(h_j, rV_k)
    """

    def __init__(self, T: int, d_in: int, dR: int):
        super().__init__()
        self.T = T
        self.d_in = d_in
        self.dR = dR

        # Parameters to be set/learned:
        self.WA = nn.Parameter(torch.zeros(d_in, d_in))  # [d_in, d_in]
        self.RA = nn.Parameter(torch.zeros(T, d_in))  # [T+1, d_in] index k in 1..T
        self.RV = nn.Parameter(torch.zeros(T, dR))  # [T+1, dR] index k in 1..T

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H: [B, T, d_in]
        Returns:
          Hhat: [B, T, d_in + dR]
          A:    [B, T, T] attention weights
        """
        B, T, d = H.shape
        assert T == self.T and d == self.d_in

        device = H.device
        allow = causal_mask(T, device=device)  # [T,T] bool

        # Content-content term: (H WA) · H^T
        HW = H @ self.WA  # [B,T,d]
        e_cc = torch.matmul(HW, H.transpose(1, 2))  # [B,T,T]

        # Content-position term: h_i^T rA_{k}, k=(i-j)+1 in 1..T
        i = torch.arange(T, device=device)[:, None]
        j = torch.arange(T, device=device)[None, :]
        k = i - j
        k = k.clamp(min=0, max=self.T - 1)

        rA = self.RA[k]  # [T,T,d]
        e_cp = (H[:, :, None, :] * rA[None, :, :, :]).sum(dim=-1)  # [B,T,T]

        e = e_cc + e_cp

        # Mask future
        e = e.masked_fill(~allow[None, :, :], float("-inf"))

        A = torch.softmax(e, dim=-1)  # [B,T,T]

        # Values: concat(h_j, rV_k)
        rV = self.RV[k]  # [T,T,dR]
        V = torch.cat(
            [
                H[:, None, :, :].expand(B, T, T, d),
                rV[None, :, :, :].expand(B, T, T, self.dR),
            ],
            dim=-1,
        )
        Hhat = torch.einsum("bij,bijd->bid", A, V)  # [B,T,d+dR]
        return Hhat, A


class DisentangledLayer(nn.Module):
    """
    One disentangled layer:
      Hhat = head(H)
      Hout = Concat(H, Hhat)  (no MLP, no additive residual)
    """

    def __init__(self, T: int, d_in: int, dR: int):
        super().__init__()
        self.head = DisentangledSingleHead(T=T, d_in=d_in, dR=dR)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Hhat, A = self.head(H)
        Hout = torch.cat([H, Hhat], dim=-1)
        return Hout, A


class ConstructedMDDisentangledTransformer(nn.Module):
    """
    Implements the *constructed* 3-layer disentangled transformer from the paper (Prop 3).

    Inputs:
      tokens y: [B,T] int64 in {0..q-1}
    Internals:
      initial embedding is one-hot e_{y_t} in R^q (so d0=q).
    Output:
      logits: [B,q] for next token (position T)
      aux dict includes attention maps and intermediate tensors.
    """

    def __init__(
        self,
        q: int,
        m: int,
        T: int,
        pi_star: torch.Tensor,
        beta: float = 10.0,
        delta1: float = 20.0,
        delta2: float = 20.0,
        delta3: float = 20.0,
        eps: float = 1e-12,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert pi_star.shape == (q, q)
        self.q = q
        self.m = m
        self.T = T
        self.eps = eps

        # store pi_star (fixed) and its log
        self.register_buffer("pi_star", row_stochastic(pi_star.to(dtype=dtype)))
        self.register_buffer("log_pi_star", torch.log(self.pi_star.clamp_min(eps)))

        # Choose dR = m as in construction (needs >= m).
        self.dR = m

        # Build 3 disentangled layers
        # Layer1 input d0=q
        self.l1 = DisentangledLayer(T=T, d_in=q, dR=self.dR)
        d1 = q + (q + self.dR)  # concat(H, Hhat)
        self.l2 = DisentangledLayer(T=T, d_in=d1, dR=self.dR)
        d2 = d1 + (d1 + self.dR)
        self.l3 = DisentangledLayer(T=T, d_in=d2, dR=self.dR)
        d3 = d2 + (d2 + self.dR)
        self.dims = {"d0": q, "d1": d1, "d2": d2, "d3": d3}

        # Output projection W_O: [q, d3] (fixed in construction)
        self.WO = nn.Parameter(torch.zeros(q, d3, dtype=dtype), requires_grad=False)

        # Set constructed parameters for all layers:
        self._init_constructed_params(
            beta=beta, delta1=delta1, delta2=delta2, delta3=delta3
        )

    def _init_constructed_params(
        self, beta: float, delta1: float, delta2: float, delta3: float
    ) -> None:
        q, m, T = self.q, self.m, self.T
        d0, d1, d2, d3 = (
            self.dims["d0"],
            self.dims["d1"],
            self.dims["d2"],
            self.dims["d3"],
        )
        dR = self.dR

        # ----- Layer 1 (posterior responsibilities) -----
        # WA = log(pi_star)
        with torch.no_grad():
            self.l1.head.WA.copy_(self.log_pi_star)

            # RA[k] = +delta1 * 1 for k in 2..m+1 (lags 1..m), else -delta1 * 1
            # Remember k = (i-j)+1. Lag g = i-j => k=g+1.
            RA1 = torch.full(
                (T, d0),
                -delta1,
                dtype=self.log_pi_star.dtype,
                device=self.log_pi_star.device,
            )
            for g in range(1, m + 1):
                k = g
                if 0 <= k < T:
                    RA1[k, :] = +delta1
            self.l1.head.RA.copy_(RA1)

            # RV[k] stores one-hot of lag g in R^m at k=g+1, else 0
            RV1 = torch.zeros(
                (T, dR),
                dtype=self.log_pi_star.dtype,
                device=self.log_pi_star.device,
            )
            for g in range(1, m + 1):
                k = g
                if 0 <= k < T:
                    RV1[k, g - 1] = 1.0
            self.l1.head.RV.copy_(RV1)

        # ----- Layer 2 (average/sum responsibilities over positions m+1..T) -----
        # WA = 0, RV = 0
        with torch.no_grad():
            self.l2.head.WA.zero_()
            self.l2.head.RV.zero_()

            # RA2[k] = 0 for k in 1..(T-m), else [-delta2*1_q, 0]
            # This makes final token attend uniformly to keys j in [m+1..T],
            # because for query i=T, keys j<=m correspond to k>=T-m+1 which get penalized.
            RA2 = torch.zeros(
                (T, d1),
                dtype=self.log_pi_star.dtype,
                device=self.log_pi_star.device,
            )
            for k in range(0, T):
                if k >= (T - m):
                    v = torch.zeros(
                        d1, dtype=self.log_pi_star.dtype, device=self.log_pi_star.device
                    )
                    v[:q] = -delta2
                    RA2[k] = v
            self.l2.head.RA.copy_(RA2)

        # ----- Layer 3 (softmax over lags to form λ̃) -----
        with torch.no_grad():
            self.l3.head.WA.zero_()
            self.l3.head.RV.zero_()

            # Need to place beta * e_g aligned with the "averaged responsibilities Γ" inside h^(2)_T
            # Locate Γ indices inside layer-3 input H^(2)_T.
            #
            # H1 = [H0(q), Hhat1(q+m)] => d1=2q+m
            # H2 = [H1(d1), Hhat2(d1+m)] => d2=2d1+m
            #
            # In Hhat1, responsibilities sit in last m dims of Hhat1 => within H1 at indices:
            #   start = q + q = 2q, length m
            #
            # Layer2 outputs Hhat2_T which averages H1_j in its first d1 dims.
            # Therefore, Γ (averaged responsibilities) sit in Hhat2_T at the same indices inside that first d1 block:
            #   within Hhat2_T: indices [2q : 2q+m)
            #
            # In H2_T = concat(H1_T, Hhat2_T): Γ sits at offset d1 + 2q ... d1+2q+m-1.
            gamma_start = d1 + (2 * q)
            gamma_end = gamma_start + m

            assert gamma_end <= d2, (gamma_start, gamma_end, d2)

            RA3 = torch.zeros(
                (T, d2),
                dtype=self.log_pi_star.dtype,
                device=self.log_pi_star.device,
            )

            # For k corresponding to active lags (g=1..m), i=T, j=T-g => k=g+1 in [2..m+1]:
            # RA3[k] = [+delta3 on first q dims, beta * e_g on Γ-block]
            for g in range(1, m + 1):
                k = g
                if 0 <= k < T:
                    v = torch.zeros(
                        d2, dtype=self.log_pi_star.dtype, device=self.log_pi_star.device
                    )
                    v[:q] = +delta3
                    v[gamma_start + (g - 1)] = beta
                    RA3[k] = v

            # All other k get [-delta3 on first q dims] to suppress attention outside lag band
            for k in range(0, T):
                if not (1 <= k <= m):
                    v = torch.zeros(
                        d2, dtype=self.log_pi_star.dtype, device=self.log_pi_star.device
                    )
                    v[:q] = -delta3
                    RA3[k] = v

            self.l3.head.RA.copy_(RA3)

        # ----- Output matrix WO selects the first q dims of Hhat3_T and multiplies by pi_star^T -----
        # H3 = concat(H2(d2), Hhat3(d2+m)).
        # In Hhat3_T, first d2 dims are weighted sum of H2_j. The first q dims of that are Σ_g λ̃_g e_{y_{T-g}}.
        # So in full H3_T, that block begins at offset d2.
        with torch.no_grad():
            self.WO.zero_()
            self.WO[:, d2 : d2 + q] = self.pi_star.transpose(0, 1)  # pi_star^T

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        y: [B,T] integer tokens
        Returns:
          logits: [B,q]
          aux: dict with attention maps and intermediate
        """
        B, T = y.shape
        assert T == self.T
        device = y.device

        # One-hot embedding: H0 [B,T,q]
        H = F.one_hot(y, num_classes=self.q).to(self.pi_star.dtype)

        H1, A1 = self.l1(H)
        H2, A2 = self.l2(H1)
        H3, A3 = self.l3(H2)

        # take last position T-1
        h_last = H3[:, -1, :]  # [B,d3]
        logits = h_last @ self.WO.transpose(0, 1)  # [B,q]
        aux = {
            "A1": A1,
            "A2": A2,
            "A3": A3,
            "H1_last": H1[:, -1, :],
            "H2_last": H2[:, -1, :],
            "H3_last": h_last,
        }
        return logits, aux


# -----------------------------
# Standard Transformer (for comparison; not trained)
# -----------------------------


class StandardTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # MultiheadAttention expects attn_mask where True means disallow in newer versions;
        # We'll build float mask with -inf for disallowed.
        out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + out)
        x = self.ln2(x + self.ff(x))
        return x


class StandardTransformer(nn.Module):
    def __init__(
        self,
        q: int,
        T: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        self.q = q
        self.T = T
        self.emb = nn.Embedding(q, d_model)
        self.pos = nn.Embedding(T, d_model)
        self.blocks = nn.ModuleList(
            [StandardTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, q)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        B, T = y.shape
        assert T == self.T
        pos_ids = torch.arange(T, device=y.device)[None, :].expand(B, T)
        x = self.emb(y) + self.pos(pos_ids)

        # Causal mask for MultiheadAttention: float mask [T,T] with -inf above diagonal
        mask = torch.full((T, T), float("-inf"), device=y.device)
        mask = torch.triu(mask, diagonal=1)  # upper triangular (disallowed)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        x = self.ln(x)
        logits = self.out(x[:, -1, :])  # next-token logits from last position
        return logits


# -----------------------------
# Demo / sanity check
# -----------------------------


@torch.no_grad()
def demo(device: str = "cpu") -> None:
    set_seed(0)
    device = torch.device(device)

    q = 7
    m = 3
    T = 64
    cfg = MTDConfig(q=q, m=m, T=T)

    pi_star = sample_pi_star(q, concentration=1.0, device=device)
    y, lam_true = generate_mtd_sequence(cfg, pi_star, device=device)

    # Direct MD estimator and implied predictive distribution
    beta = 12.0
    lam_md = one_step_md_lambda_hat(y, pi_star, m=m, eta_or_beta=beta, average=True)
    p_md = mtd_predict_next_from_lambda(y, pi_star, lam_md, m=m)

    # Constructed transformer
    model = ConstructedMDDisentangledTransformer(
        q=q,
        m=m,
        T=T,
        pi_star=pi_star,
        beta=beta,
        delta1=30.0,
        delta2=30.0,
        delta3=30.0,
        dtype=torch.float32,
    ).to(device)
    model.eval()

    logits, aux = model(y[None, :])
    T, m = cfg.T, cfg.m
    A3_last = aux["A3"][0, -1]  # [T]

    print("A3_last mass on lag keys:")
    for g in range(1, m + 1):
        print(g, "pos", T - g, "A3", float(A3_last[T - g]))

    print("A3_last total mass on lag band:", float(A3_last[T - m : T].sum()))
    print("A3_last total mass elsewhere:", float(A3_last[: T - m].sum()))
    p_tf = torch.softmax(logits[0], dim=-1)

    # Extract the layer-3 attention weights for the last query position:
    # they should place λ̃_g at keys j=T-g (and near-zero elsewhere), in the large-bias regime.
    A3_last = aux["A3"][0, -1]  # [T]
    lam_from_attn = torch.stack([A3_last[T - g] for g in range(1, m + 1)], dim=0)

    print("=== Mixture weights ===")
    print("true λ          :", lam_true.cpu())
    print("MD  λ̂ (avg form):", lam_md.cpu())
    print("attn-derived λ̃  :", lam_from_attn.cpu())
    print("L1 distance MD vs attn:", (lam_md - lam_from_attn).abs().sum().item())

    print("\n=== Predictive distribution p(next) ===")
    print(
        "KL(p_MD || p_tf):",
        torch.sum(
            p_md * (p_md.clamp_min(1e-12).log() - p_tf.clamp_min(1e-12).log())
        ).item(),
    )
    print("Top-5 tokens MD:", torch.topk(p_md, k=min(5, q)))
    print("Top-5 tokens TF:", torch.topk(p_tf, k=min(5, q)))


if __name__ == "__main__":
    demo("cpu")
    # demo("cuda")  # if available
