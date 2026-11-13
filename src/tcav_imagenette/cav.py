"""CAV dataset construction, linear CAV classifier, training/validation, orientation, unit-norm directions.

A CAV is trained in the *raw* (pooled) activation space of its layer: logit(x) = w^T (f(x) - mu) + b.
Centering by the CAV-training mean only shifts the bias, so w is the normal of the decision
hyperplane in the original feature space (the space the gradients live in). Label 1 = concept
examples, so moving along +w increases concept presence. The CAV is v = w / ||w||_2.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KIND_ID, Config


class LinearCAV(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (N,d) -> (N,)
        return self.linear(x).squeeze(-1)


def make_cav_dataset(layer: str, concept: str, kind: str, rng: np.random.Generator,
                     concept_feats: Dict[str, Dict[str, torch.Tensor]], ref_feats: Dict[str, torch.Tensor],
                     n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Balanced binary dataset in feature space. Label 1 = 'concept side'. Returns X (2n,C), y (2n,).

    kind == 'concept'          : n concept images (label 1) vs n random reference images (label 0)
    kind == 'shuffled_labels'  : same features as 'concept' but the labels randomly permuted (balanced)
    kind == 'random_reference' : n random reference images (label 1) vs another n (label 0)
    """
    R = ref_feats[layer]
    ones_zeros = torch.cat([torch.ones(n), torch.zeros(n)]).to(R.device)
    if kind in ("concept", "shuffled_labels"):
        pos = concept_feats[layer][concept]
        assert pos.shape[0] == n
        neg_idx = torch.as_tensor(rng.choice(R.shape[0], size=n, replace=False), device=R.device)
        X = torch.cat([pos, R[neg_idx]])
        y = ones_zeros
        if kind == "shuffled_labels":                      # random labels, same pool, same class balance
            y = y[torch.as_tensor(rng.permutation(2 * n), device=R.device)]
    elif kind == "random_reference":                       # concept-independent: reference vs reference
        idx = torch.as_tensor(rng.choice(R.shape[0], size=2 * n, replace=False), device=R.device)
        X, y = R[idx], ones_zeros
    else:
        raise ValueError(kind)
    assert X.shape[0] == y.shape[0] == 2 * n and int(y.sum().item()) == n   # balanced
    return X, y


def stratified_split(y_np: np.ndarray, val_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class random split into (train_idx, val_idx)."""
    tr, va = [], []
    for cls in (0, 1):
        idx = rng.permutation(np.where(y_np == cls)[0])
        nv = int(round(val_frac * len(idx)))
        va += idx[:nv].tolist()
        tr += idx[nv:].tolist()
    return np.array(tr), np.array(va)


def train_linear_cav(X_tr: torch.Tensor, y_tr: torch.Tensor, X_va: torch.Tensor, y_va: torch.Tensor,
                     lr: float, steps: int, l2: float) -> Tuple[torch.Tensor, float, float]:
    """Full-batch Adam on logistic loss + L2 (0.5 * l2 * ||w||^2). Features are centred with the
    CAV-train mean only. Returns the raw weight vector w (d,), train accuracy, validation accuracy."""
    mu = X_tr.mean(dim=0, keepdim=True)                     # train-split statistics only (no leakage)
    Xtr, Xva = X_tr - mu, X_va - mu
    clf = LinearCAV(X_tr.shape[1]).to(X_tr.device)
    nn.init.zeros_(clf.linear.weight)
    nn.init.zeros_(clf.linear.bias)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(clf(Xtr), y_tr) + 0.5 * l2 * clf.linear.weight.pow(2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc_tr = ((clf(Xtr) > 0).float() == y_tr).float().mean().item()
        acc_va = ((clf(Xva) > 0).float() == y_va).float().mean().item()
        w = clf.linear.weight.detach().squeeze(0).clone()
    return w, acc_tr, acc_va


def unit_normalize(w: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, float, bool]:
    """Return (unit vector, ||w||, finite flag). Non-finite / (near-)zero w gives a zero vector."""
    w_norm = w.norm().item()
    finite = bool(torch.isfinite(w).all().item()) and w_norm > eps
    v = w / w_norm if finite else torch.zeros_like(w)
    return v, w_norm, finite


def check_orientation(v: torch.Tensor, X_va: torch.Tensor, y_va: torch.Tensor, finite: bool = True) -> bool:
    """True if validation examples of the positive class project *higher* on v than negative ones,
    i.e. v points toward increasing concept presence."""
    if not finite:
        return False
    proj = X_va @ v
    return bool((proj[y_va == 1].mean() > proj[y_va == 0].mean()).item())


def fit_cav_run(layer: str, concept: str, concept_idx: int, run: int, kind: str,
                concept_feats: Dict[str, Dict[str, torch.Tensor]], ref_feats: Dict[str, torch.Tensor],
                cfg: Config) -> dict:
    """One CAV run: balanced dataset -> stratified split -> linear classifier -> unit CAV -> checks.

    The RNG seed depends on (concept, run, kind) but NOT on the layer, so every layer sees the same
    images and split. A CAV is ``valid`` if finite and correctly oriented and (concept CAVs only)
    validation accuracy >= ``cfg.cav_min_acc``. Control CAVs are only required to be finite.
    """
    rng = np.random.default_rng([cfg.seed, concept_idx, run, KIND_ID[kind]])
    X, y = make_cav_dataset(layer, concept, kind, rng, concept_feats, ref_feats, cfg.n_concept_imgs)
    tr, va = stratified_split(y.cpu().numpy(), cfg.cav_val_frac, rng)
    tr_t, va_t = torch.as_tensor(tr, device=X.device), torch.as_tensor(va, device=X.device)
    w, acc_tr, acc_va = train_linear_cav(X[tr_t], y[tr_t], X[va_t], y[va_t],
                                         lr=cfg.cav_lr, steps=cfg.cav_steps, l2=cfg.cav_l2)

    v, w_norm, finite = unit_normalize(w)                   # unit-norm CAV, points toward label-1 side
    orient_ok = check_orientation(v, X[va_t], y[va_t], finite)
    if kind == "concept":
        valid = finite and orient_ok and acc_va >= cfg.cav_min_acc
    else:
        valid = finite
    return {"v": v.detach().cpu(), "train_acc": acc_tr, "val_acc": acc_va, "w_norm": w_norm,
            "finite": finite, "orient_ok": orient_ok, "valid": valid}
