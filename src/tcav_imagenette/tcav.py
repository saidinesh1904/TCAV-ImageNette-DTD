"""TCAV: layer-specific intervention, autograd directional derivatives, finite-difference check, scores.

For class k, pooled feature p = f(x) = GAP(A) and CAV v (unit norm, trained in the SAME pooled space):

    S_{C,k}(x) = grad_p h_k(p) . v = d/d(eps) h_k(p + eps v) |_{eps=0}

where h_k is the *logit* of class k computed by the real downstream network.

Implementation (``forward_from_pooled``): the downstream network is evaluated on ``A + (p - GAP(A))``
where A is the detached activation map and p is a leaf tensor equal to GAP(A). The added shift is zero
in value (logits equal ``model(x)``), but the graph passes through p. Perturbing p by delta shifts every
spatial position of each channel by delta, changing the pooled feature by exactly delta:

* layer4: downstream is GAP + linear head, so grad_p h_k = w_k (the same for every image);
* layer3: downstream contains the non-linear layer4 block, so the derivative is defined through an
  explicit intervention, a spatially uniform per-channel shift of the layer3 activation map.

Because grad_p h_k does not depend on v, it is computed once per image; S for any CAV is a dot product.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .config import LAYER_DIMS, NUM_CLASSES, Config
from .data import ImagePathDataset, build_transforms, make_loader
from .features import SplitResNet


# --------------------------------------------------------------------------- intervention
def forward_from_pooled(split: SplitResNet, A: Tensor, p: Tensor, layer: str) -> Tensor:
    """A: detached activation map (B,C,H,W); p: pooled feature (B,C) (may require grad), p == GAP(A) in value.
    Returns logits (B,10); gradient w.r.t. p flows through the REAL downstream network."""
    shift = (p - A.mean(dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)       # (B,C,1,1), zero in value
    return split.downstream(A + shift, layer)


def pooled_gradient(split: SplitResNet, A: Tensor, k: int, layer: str) -> Tuple[Tensor, Tensor]:
    """grad_p h_k for a batch of activation maps. Returns (grad (B,C), logits (B,10))."""
    p = A.mean(dim=(2, 3)).clone().requires_grad_(True)               # (B,C) leaf = f(x)
    logits = forward_from_pooled(split, A, p, layer)                  # (B,10)
    (g,) = torch.autograd.grad(logits[:, k].sum(), p)                 # samples independent (eval-mode BN)
    return g.detach(), logits.detach()


def compute_class_gradients(split: SplitResNet, paths: Sequence[str], k: int, layers: Sequence[str],
                            cfg: Config, device: torch.device, eval_tf=None
                            ) -> Tuple[Dict[str, Tensor], Tensor]:
    """For images of ground-truth class k: gradient of logit h_k w.r.t. the pooled feature f(x), for
    every layer. Returns ({layer: (n, C) tensor on CPU}, predictions (n,))."""
    if eval_tf is None:
        _, eval_tf = build_transforms(cfg)
    loader = make_loader(ImagePathDataset(paths, np.full(len(paths), k), eval_tf), cfg.tcav_batch_size, cfg, device)
    grads = {l: [] for l in layers}
    preds = []
    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            acts = split.trunk_multi(x, layers)                       # activation maps, no graph
        for i, l in enumerate(layers):
            g, logits = pooled_gradient(split, acts[l], k, l)
            grads[l].append(g.cpu())
            if i == 0:
                preds.append(logits.argmax(1).cpu())
    return {l: torch.cat(v) for l, v in grads.items()}, torch.cat(preds)


def compute_all_gradients(split: SplitResNet, test_paths_by_class: Dict[int, Sequence[str]], cfg: Config,
                          device: torch.device, progress: bool = True):
    """Gradients for all ImageNette classes. Returns (GRADS[layer][k] -> (n_k, C), TEST_PRED[k], CORRECT_MASK[k])."""
    from tqdm.auto import tqdm
    _, eval_tf = build_transforms(cfg)
    grads = {l: {} for l in cfg.tcav_layers}
    test_pred, correct_mask = {}, {}
    for k in tqdm(range(NUM_CLASSES), desc="class gradients", disable=not progress):
        gk, pk = compute_class_gradients(split, test_paths_by_class[k], k, cfg.tcav_layers, cfg, device, eval_tf)
        for l in cfg.tcav_layers:
            grads[l][k] = gk[l]
        test_pred[k] = pk
        correct_mask[k] = (pk == k)
    return grads, test_pred, correct_mask


def check_gradients(grads: Dict[str, Dict[int, Tensor]], test_paths_by_class: Dict[int, Sequence[str]],
                    layers: Sequence[str]) -> Dict[str, float]:
    """Shape / finiteness / non-zero checks (raise AssertionError) and, per layer, the mean relative
    std of the gradient across images (~0 means an image-independent gradient, e.g. layer4)."""
    for l in layers:
        for k in range(NUM_CLASSES):
            G = grads[l][k]
            assert G.shape == (len(test_paths_by_class[k]), LAYER_DIMS[l]), (l, k, G.shape)
            assert torch.isfinite(G).all(), f"non-finite gradients ({l}, class {k})"
            assert G.abs().sum() > 0, f"all-zero gradients ({l}, class {k}) -> disconnected graph?"
    out = {}
    for l in layers:
        rel = [(grads[l][k].std(dim=0).mean() / (grads[l][k].abs().mean() + 1e-12)).item()
               for k in range(NUM_CLASSES)]
        out[l] = float(np.mean(rel))
    return out


# --------------------------------------------------------------------------- derivative checks
def directional_derivative_autograd(split: SplitResNet, A: Tensor, k: int, v: Tensor, layer: str) -> Tensor:
    """S = d/d(eps) h_k(p + eps v) at eps=0, differentiating w.r.t. a scalar eps per image. Returns (B,)."""
    vd = v.to(A.device).view(1, -1)
    p0 = A.mean(dim=(2, 3))
    e = torch.zeros(A.shape[0], 1, device=A.device, dtype=A.dtype, requires_grad=True)
    (s,) = torch.autograd.grad(forward_from_pooled(split, A, p0 + e * vd, layer)[:, k].sum(), e)
    return s.squeeze(1)


@torch.no_grad()
def directional_derivative_fd(split: SplitResNet, A: Tensor, k: int, v: Tensor, layer: str,
                              eps: float = 1e-2) -> Tensor:
    """Central finite difference (h_k(p + eps v) - h_k(p - eps v)) / (2 eps). Returns (B,)."""
    vd = v.to(A.device).view(1, -1)
    p0 = A.mean(dim=(2, 3))
    hp = forward_from_pooled(split, A, p0 + eps * vd, layer)[:, k]
    hm = forward_from_pooled(split, A, p0 - eps * vd, layer)[:, k]
    return (hp - hm) / (2 * eps)


def verify_directional_derivative(split: SplitResNet, x: Tensor, layer: str, k: int, v: Tensor,
                                  stored_grads: Optional[Tensor] = None, eps: float = 1e-2) -> Dict[str, float]:
    """Compare ways to get S_{C,k}(x) on the images ``x`` (already preprocessed, on the model device):
       (a) autograd gradient . v, (b) autograd d/d(eps) h_k(f + eps v), (c) central finite differences,
       (d) stored gradients . v (if ``stored_grads`` for the same images is given).
    Returns max absolute differences and the scale of S."""
    with torch.no_grad():
        A = split.trunk(x, layer)
    g, _ = pooled_gradient(split, A, k, layer)
    s_a = (g * v.to(A.device).view(1, -1)).sum(1)
    s_b = directional_derivative_autograd(split, A, k, v, layer)
    s_c = directional_derivative_fd(split, A, k, v, layer, eps)
    res = {"max_abs_a_b": float((s_a - s_b).abs().max()),
           "max_abs_a_c": float((s_a - s_c).abs().max()),
           "scale_S": float(s_a.abs().mean())}
    if stored_grads is not None:
        s_d = (stored_grads[: x.shape[0]] @ v.cpu()).to(A.device)
        res["max_abs_a_d"] = float((s_a - s_d).abs().max())
    return res


# --------------------------------------------------------------------------- TCAV score
def summarize_derivatives(S: Tensor) -> dict:
    """TCAV score = fraction of examples with strictly positive directional derivative
    (exact zeros count as non-positive)."""
    n = int(S.numel())
    if n == 0:
        return {"n_images": 0, "n_pos": 0, "n_neg": 0, "tcav": np.nan, "frac_neg": np.nan,
                "mean_S": np.nan, "std_S": np.nan}
    return {"n_images": n,
            "n_pos": int((S > 0).sum()), "n_neg": int((S < 0).sum()),
            "tcav": float((S > 0).float().mean()),               # TCAV score
            "frac_neg": float((S < 0).float().mean()),
            "mean_S": float(S.mean()), "std_S": float(S.std()) if n > 1 else 0.0}


def tcav_for_cav(grads: Dict[str, Dict[int, Tensor]], correct_mask: Dict[int, Tensor], layer: str,
                 v: Tensor, k: int, only_correct: bool = False) -> dict:
    """S_{C,k}(x) = grad_{f(x)} h_k . v for all x in X_k (gradients computed by autograd).
    X_k = all ground-truth class-k test images, or only the correctly classified ones if requested."""
    G = grads[layer][k]                                            # (n_k, C)
    if only_correct:
        G = G[correct_mask[k]]
    S = G @ v.cpu().float()                                        # (n,) directional derivatives
    return summarize_derivatives(S)
