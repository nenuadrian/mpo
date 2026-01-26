# mpo_transformer_bandit.py
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mpo_step_full_actions(
    policy: nn.Module,
    opt: torch.optim.Optimizer,
    eta_param: torch.Tensor,
    bandit,
    batch_size: int,
    eps_kl: float,
    device: str,
    eta_lr: float = 1e-3,
):
    """
    MPO-like update for small discrete action spaces:
      - compute q*(a|x) over ALL actions (no sampling)
      - M-step: cross-entropy with soft targets q*
      - eta update: drive KL(q*||pi_old) toward eps_kl
    """
    x, target = bandit.sample_batch(batch_size, device=device)
    A = bandit.cfg.n_actions

    # old policy (stop-grad)
    logits_old = policy(x).detach()
    logpi_old = F.log_softmax(logits_old, dim=-1)  # [B, A]
    pi_old = logpi_old.exp()

    # Q values for all actions: Q=1 if action matches target else 0 (plus noise not applied here)
    # If you want reward noise, you can inject it into Q too, but deterministic Q makes learning clearer.
    actions = torch.arange(A, device=device).view(1, A).expand(batch_size, A)  # [B, A]
    Q = (actions == target.view(-1, 1)).float()  # [B, A] in {0,1}

    # positive eta
    eta = F.softplus(eta_param) + 1e-6

    # E-step: q*(a|x) ∝ pi_old(a|x) * exp(Q/eta)
    logq_unnorm = logpi_old + Q / eta
    q = F.softmax(logq_unnorm, dim=-1)  # [B, A]

    # compute KL(q || pi_old) (this is the trust-region statistic)
    kl = (q * (torch.log(q + 1e-12) - logpi_old)).sum(dim=-1).mean()

    # M-step: minimize KL(q || pi_theta) => cross-entropy with soft targets q
    logits = policy(x)
    logpi = F.log_softmax(logits, dim=-1)
    policy_loss = -(q.detach() * logpi).sum(dim=-1).mean()

    # eta update: push KL toward eps_kl (stable squared error)
    # do this as part of the same backward pass by adding a penalty
    eta_loss = (kl - eps_kl).pow(2)

    loss = policy_loss + eta_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    # diagnostics
    with torch.no_grad():
        x2, target2 = bandit.sample_batch(batch_size, device=device)
        pred = policy(x2).argmax(dim=-1)
        acc = (pred == target2).float().mean().item()
        eta_val = (F.softplus(eta_param) + 1e-6).item()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "eta_loss": eta_loss.item(),
        "eta": eta_val,
        "kl": kl.item(),
        "acc": acc,
    }


# -----------------------------
# Toy contextual bandit
# -----------------------------
@dataclass
class BanditConfig:
    vocab_size: int = 32  # tokens in context
    ctx_len: int = 16  # sequence length
    n_actions: int = 8  # discrete actions
    noise_p: float = 0.05  # flips reward with this prob


class ContextualBandit:
    """
    Observation: x in {0..vocab_size-1}^{ctx_len}
    Hidden target rule: a*(x) = (sum(x) + x[0]*3) mod n_actions
    Reward: 1 if a == a*(x) else 0, with optional noise.
    """

    def __init__(self, cfg: BanditConfig):
        self.cfg = cfg

    def sample_batch(self, batch_size: int, device: str):
        x = torch.randint(
            low=0,
            high=self.cfg.vocab_size,
            size=(batch_size, self.cfg.ctx_len),
            device=device,
        )
        target = (x.sum(dim=1) + 3 * x[:, 0]) % self.cfg.n_actions
        return x, target

    def reward(self, target_action: torch.Tensor, a: torch.Tensor):
        # r = 1[a == target], with label noise
        r = (a == target_action).float()
        if self.cfg.noise_p > 0:
            flip = (torch.rand_like(r) < self.cfg.noise_p).float()
            r = torch.abs(r - flip)  # flip 0<->1
        return r


# -----------------------------
# Tiny causal Transformer policy
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        qkv = self.qkv(x)  # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, H, T, Dh]
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # causal mask
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, H, T, T]
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        w = F.softmax(att, dim=-1)
        y = w @ v  # [B, H, T, Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ctx_len: int,
        n_actions: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, ctx_len, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_actions)

    def forward(self, x_tokens):
        # x_tokens: [B, T]
        B, T = x_tokens.shape
        h = self.tok_emb(x_tokens) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        # Use last token representation as summary for action
        logits = self.head(h[:, -1, :])  # [B, A]
        return logits


# -----------------------------
# MPO-style training loop
# -----------------------------
def mpo_step(
    policy: nn.Module,
    opt: torch.optim.Optimizer,
    eta_param: torch.Tensor,
    bandit: ContextualBandit,
    batch_size: int,
    K: int,
    eps_kl: float,
    device: str,
):
    """
    One MPO update:
      - sample contexts x, targets
      - compute pi_old(a|x)
      - sample K actions per context from pi_old
      - get rewards r_k = Q_k (one-step)
      - E-step: q_k ∝ pi_old(a_k|x) * exp(r_k / eta)
      - update eta via dual objective g(eta)
      - M-step: minimize KL(q || pi_theta) over sampled support (weighted log-probs)
    """
    x, target = bandit.sample_batch(batch_size, device=device)

    logits_old = policy(x).detach()  # stop-gradient "old" policy
    logp_old = F.log_softmax(logits_old, dim=-1)  # [B, A]
    p_old = logp_old.exp()

    # Sample K actions per context: [B, K]
    a = torch.multinomial(p_old, num_samples=K, replacement=True)

    # Gather log pi_old(a_k|x): [B, K]
    logp_old_ak = logp_old.gather(dim=1, index=a)

    # Rewards (Q estimates): [B, K]
    r = bandit.reward(target.unsqueeze(1).expand_as(a), a)

    # Keep eta positive via softplus
    eta = F.softplus(eta_param) + 1e-6

    # ----- E-step: construct nonparametric q over sampled actions -----
    # unnormalized log q_k = log pi_old(a_k|x) + r_k / eta
    logq_unnorm = logp_old_ak + r / eta
    q = F.softmax(logq_unnorm, dim=1)  # [B, K], sums to 1 across sampled actions

    # ----- Dual objective for eta (sample-based) -----
    # g(eta) = eta*eps + eta*log( (1/K) sum_k exp(r_k / eta) )
    # Here we ignore logpi_old term in the expectation because our q already includes it.
    # This keeps the demo simple while still behaving like "temperature from KL target".
    # More faithful variants include the full partition function involving pi_old.
    log_mean_exp = torch.logsumexp(r / eta, dim=1) - math.log(K)  # [B]
    dual = (eta * eps_kl + eta * log_mean_exp).mean()

    # ----- M-step policy loss -----
    logits = policy(x)
    logp = F.log_softmax(logits, dim=-1)
    logp_ak = logp.gather(dim=1, index=a)  # [B, K]
    # Minimize KL(q || pi_theta) over sampled support => maximize sum_k q_k log pi(a_k|x)
    policy_loss = -(q.detach() * logp_ak).sum(dim=1).mean()

    loss = policy_loss + dual  # joint update of theta and eta

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    # Diagnostics
    with torch.no_grad():
        # Approx KL(q || pi_old_on_support) using sampled support distribution proportional to pi_old(a_k|x)
        # pi_old_on_support normalized over the sampled set:
        pi_old_sup = F.softmax(logp_old_ak, dim=1)
        kl_est = (
            (q * (torch.log(q + 1e-9) - torch.log(pi_old_sup + 1e-9)))
            .sum(dim=1)
            .mean()
            .item()
        )

        # Greedy accuracy on a fresh minibatch
        x2, target2 = bandit.sample_batch(batch_size, device=device)
        pred = policy(x2).argmax(dim=-1)
        acc = (pred == target2).float().mean().item()

        eta_val = (F.softplus(eta_param) + 1e-6).item()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "dual": dual.item(),
        "eta": eta_val,
        "kl_est_support": kl_est,
        "greedy_acc": acc,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)

    cfg = BanditConfig(vocab_size=32, ctx_len=16, n_actions=8, noise_p=0.05)
    bandit = ContextualBandit(cfg)

    policy = TransformerPolicy(
        vocab_size=cfg.vocab_size,
        ctx_len=cfg.ctx_len,
        n_actions=cfg.n_actions,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
    ).to(device)

    # eta parameter in unconstrained space
    eta_param = nn.Parameter(torch.tensor(1.0, device=device))
    opt = torch.optim.Adam(list(policy.parameters()) + [eta_param], lr=3e-4)

    batch_size = 256
    K = 16
    eps_kl = 0.05  # target KL bound (small => conservative updates)

    for step in range(1, 20001):
        stats = mpo_step_full_actions(
            policy, opt, eta_param, bandit, batch_size, eps_kl, device
        )
        if step % 200 == 0:
            print(
                f"step {step:4d} | loss {stats['loss']:.4f} | "
                f"eta {stats['eta']:.4f} | kl {stats['kl']:.4f}"
            )


if __name__ == "__main__":
    main()
