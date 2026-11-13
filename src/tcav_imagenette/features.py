"""Intermediate feature extraction (layer3 / layer4, global average pooling) and SplitResNet."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .config import LAYER_DIMS, LAYER_ORDER, Config
from .data import ImagePathDataset, build_transforms, make_loader


class SplitResNet:
    """ResNet50 forward pass split at a residual stage (eval mode, BatchNorm uses running statistics).

    * ``trunk(x, layer)``   : image -> activation map A (B, C, H, W) after ``layer``;
      the pooled feature is f(x) = GAP(A) (layer4: C=2048, layer3: C=1024).
    * ``downstream(A, layer)``: remaining residual stages -> global average pool -> fc -> logits.
    """

    def __init__(self, model: nn.Module):
        self.model = model.eval()
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.stages = OrderedDict((n, getattr(model, n)) for n in LAYER_ORDER)
        self.fc = model.fc

    def trunk_multi(self, x: torch.Tensor, layers: Sequence[str]) -> Dict[str, torch.Tensor]:
        """x: (B,3,H,W) -> {layer: activation map (B,C,h,w)}; runs the network once up to the deepest requested layer."""
        h = self.stem(x)
        out = {}
        for name, stage in self.stages.items():
            h = stage(h)
            if name in layers:
                out[name] = h
            if len(out) == len(layers):
                break
        return out

    def trunk(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        return self.trunk_multi(x, [layer])[layer]

    def downstream(self, A: torch.Tensor, layer: str) -> torch.Tensor:
        """Activation map after ``layer`` (B,C,H,W) -> logits (B,10): remaining stages, GAP, fc."""
        h, seen = A, False
        for name, stage in self.stages.items():
            if seen:
                h = stage(h)
            if name == layer:
                seen = True
        pooled = h.mean(dim=(2, 3))            # same as torchvision's AdaptiveAvgPool2d(1) + flatten -> (B,2048)
        return self.fc(pooled)                 # (B,10)


def check_split(model: nn.Module, split: SplitResNet, x: torch.Tensor, layers: Sequence[str],
                device: torch.device, atol_act: float = 1e-5, atol_logits: float = 1e-4,
                rtol_logits: float = 1e-4) -> Dict[str, dict]:
    """Shape checks + equivalence of the split network with ``model(x)`` and with forward hooks.

    Returns per-layer results; raises ``AssertionError`` on shape mismatches. Equivalence flags are
    returned (not asserted) as in the notebook, so callers decide how to react.
    """
    x = x.to(device)
    cap = {}
    hooks = [getattr(model, l).register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, o.detach()))
             for l in layers]
    with torch.no_grad():
        ref = model(x)
        acts = split.trunk_multi(x, layers)
    for h in hooks:
        h.remove()
    results = {}
    for l in layers:
        A = acts[l]
        pooled = A.mean(dim=(2, 3))
        assert A.shape[1] == LAYER_DIMS[l] and pooled.shape == (x.shape[0], LAYER_DIMS[l])
        with torch.no_grad():
            mine = split.downstream(A, l)
        results[l] = {
            "activation_shape": tuple(A.shape),
            "pooled_shape": tuple(pooled.shape),
            "matches_hook": bool(torch.allclose(A, cap[l], atol=atol_act)),
            "downstream_matches_model": bool(torch.allclose(mine, ref, atol=atol_logits, rtol=rtol_logits)),
            "max_abs_diff": float((mine - ref).abs().max().item()),
        }
    return results


@torch.no_grad()
def extract_pooled_features(split: SplitResNet, paths: Sequence[str], layers: Sequence[str], cfg: Config,
                            device: torch.device, desc: str = "features", eval_tf=None
                            ) -> Dict[str, torch.Tensor]:
    """Global-average-pooled features for a list of image paths, using the SAME eval preprocessing.
    Returns {layer: tensor (N, C)} on CPU (float32)."""
    if eval_tf is None:
        _, eval_tf = build_transforms(cfg)
    loader = make_loader(ImagePathDataset(paths, None, eval_tf), cfg.tcav_batch_size, cfg, device)
    outs = {l: [] for l in layers}
    for x, _ in tqdm(loader, desc=desc, leave=False):
        acts = split.trunk_multi(x.to(device), layers)
        for l in layers:
            outs[l].append(acts[l].mean(dim=(2, 3)).float().cpu())      # (B,C)
    res = {l: torch.cat(v) for l, v in outs.items()}
    for l in layers:
        assert res[l].shape == (len(paths), LAYER_DIMS[l]), (l, res[l].shape)
    return res


def extract_concept_and_reference_features(split: SplitResNet, concept_paths: Dict[str, List[str]],
                                           ref_paths: Sequence[str], cfg: Config, device: torch.device
                                           ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Features of all concept and reference images, extracted once (one pass gives all layers).
    Returns (concept_feats[layer][concept] (N,C), ref_feats[layer] (R,C)) on ``device``."""
    _, eval_tf = build_transforms(cfg)
    concept_feats = {l: {} for l in cfg.tcav_layers}
    for c in cfg.concepts:
        f = extract_pooled_features(split, concept_paths[c], cfg.tcav_layers, cfg, device,
                                    desc=f"concept: {c}", eval_tf=eval_tf)
        for l in cfg.tcav_layers:
            concept_feats[l][c] = f[l].to(device)
    f = extract_pooled_features(split, ref_paths, cfg.tcav_layers, cfg, device,
                                desc="reference pool", eval_tf=eval_tf)
    ref_feats = {l: f[l].to(device) for l in cfg.tcav_layers}
    return concept_feats, ref_feats


def feature_cache_fingerprint(cfg: Config, model_fingerprint: str) -> str:
    """Fingerprint everything that changes the pooled feature tensors."""
    payload = {
        "model_sha256": model_fingerprint,
        "layers": list(cfg.tcav_layers),
        "resize": cfg.resize,
        "img_size": cfg.img_size,
        "mean": list(cfg.mean),
        "std": list(cfg.std),
        "preprocessing": "resize-center-crop-to-tensor-normalize-v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_features(path: Path, concept_feats, ref_feats, concept_paths, ref_paths,
                  model_fingerprint: str, cache_fingerprint: str) -> None:
    """Save CPU feature tensors, source image paths, and model/preprocessing identity."""
    payload = {
        "concept_feats": {l: {c: t.detach().cpu() for c, t in d.items()} for l, d in concept_feats.items()},
        "ref_feats": {l: t.detach().cpu() for l, t in ref_feats.items()},
        "concept_paths": {c: list(p) for c, p in concept_paths.items()},
        "ref_paths": list(ref_paths),
        "layers": list(ref_feats.keys()),
        "model_fingerprint": model_fingerprint,
        "cache_fingerprint": cache_fingerprint,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_features(path: Path, device: torch.device, map_location: Optional[str] = "cpu",
                  expected_model_fingerprint: Optional[str] = None,
                  expected_cache_fingerprint: Optional[str] = None):
    """Load features and optionally reject stale caches from another checkpoint/preprocessing setup.

    Old caches without fingerprints are rejected when expected fingerprints are supplied.
    Returns (concept_feats, ref_feats, concept_paths, ref_paths).
    """
    p = torch.load(str(path), map_location=map_location)
    if expected_model_fingerprint is not None:
        actual = p.get("model_fingerprint")
        if actual != expected_model_fingerprint:
            raise RuntimeError("feature cache was generated from a different or unidentified model checkpoint; "
                               "rerun extract_features.py")
    if expected_cache_fingerprint is not None:
        actual = p.get("cache_fingerprint")
        if actual != expected_cache_fingerprint:
            raise RuntimeError("feature cache preprocessing/layers do not match the current configuration; "
                               "rerun extract_features.py")
    concept_feats = {l: {c: t.to(device) for c, t in d.items()} for l, d in p["concept_feats"].items()}
    ref_feats = {l: t.to(device) for l, t in p["ref_feats"].items()}
    return concept_feats, ref_feats, p["concept_paths"], p["ref_paths"]
