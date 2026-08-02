"""Small MLPs with the instrumentation the diagnostics need.

Activation statistics (dormant fraction, effective rank) are computed from a *probe batch* held
fixed across a run, so that changes in the statistic reflect the network changing rather than the
inputs changing. Recomputing them on whatever happened to be in the last rollout would confound
network collapse with a shift in state visitation, which is precisely one of the failure modes we
are trying to tell apart.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.hidden_sizes = hidden
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))

    @torch.no_grad()
    def activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Post-ReLU activations for each hidden layer."""
        acts: list[torch.Tensor] = []
        h = x
        for layer in self.body:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                acts.append(h.clone())
        return acts


@torch.no_grad()
def dormant_fraction(model: MLP, probe: torch.Tensor, tau: float = 0.025) -> float:
    """Fraction of hidden units that are dormant, following Sokar et al.

    A unit's score is its mean absolute activation divided by the layer's mean; a unit at or below
    `tau` contributes essentially nothing to the layer's output. Reported over all hidden layers.
    """
    dormant = 0
    total = 0
    for act in model.activations(probe):
        mean_abs = act.abs().mean(dim=0)  # per-unit
        layer_mean = mean_abs.mean()
        if layer_mean <= 0:
            # Whole layer is dead. Every unit counts as dormant; guard the divide.
            dormant += mean_abs.numel()
            total += mean_abs.numel()
            continue
        score = mean_abs / layer_mean
        dormant += int((score <= tau).sum().item())
        total += score.numel()
    return dormant / max(total, 1)


@torch.no_grad()
def effective_rank(model: MLP, probe: torch.Tensor, delta: float = 0.01) -> float:
    """srank_delta of the penultimate feature matrix.

    The smallest k whose top-k singular values carry (1 - delta) of the spectral mass. Collapsing
    representations show up here well before the return moves.
    """
    acts = model.activations(probe)
    if not acts:
        return float("nan")
    feats = acts[-1]
    if feats.shape[0] < 2:
        return float("nan")
    feats = feats - feats.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(feats.double())
    total = sv.sum()
    if total <= 0:
        return 0.0
    frac = torch.cumsum(sv, dim=0) / total
    k = int(torch.searchsorted(frac, torch.tensor(1.0 - delta, dtype=frac.dtype)).item()) + 1
    return float(min(k, sv.numel()))


@torch.no_grad()
def weight_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        total += float((p.detach() ** 2).sum().item())
    return float(np.sqrt(total))


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float((p.grad.detach() ** 2).sum().item())
    return float(np.sqrt(total))


def make_probe_batch(obs_dim: int, n: int, seed: int) -> torch.Tensor:
    """A fixed batch of observations used only for activation statistics.

    Drawn from a standard normal rather than from the environment: the point is a *stationary*
    input distribution, so the statistic isolates the network. It is never used for training.
    """
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.standard_normal((n, obs_dim)), dtype=torch.float32)
