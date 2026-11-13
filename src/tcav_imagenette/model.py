"""ResNet50 (10-class) initialisation, fine-tuning, evaluation, and checkpoint handling."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models
from tqdm.auto import tqdm

from .config import CLASS_NAMES, NUM_CLASSES, WNIDS, Config
from .utils import load_checkpoint, save_checkpoint

CKPT_NAME = "resnet50_imagenette_best.pt"
HIST_NAME = "training_history.json"


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """torchvision ResNet50 (ImageNet IMAGENET1K_V2 weights); 1000-way fc replaced by a 10-way linear layer."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)      # 2048 -> 10
    return m


def freeze_model(model: nn.Module) -> nn.Module:
    """Eval mode, no parameter gradients (TCAV only needs gradients w.r.t. activations)."""
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model


def _split_digest(paths, labels) -> Optional[str]:
    """Stable digest of an ordered image-path/label split (no image bytes are read)."""
    if paths is None and labels is None:
        return None
    if paths is None or labels is None or len(paths) != len(labels):
        raise ValueError("paths and labels must both be supplied and have equal lengths")
    digest = hashlib.sha256()
    for path, label in zip(paths, labels):
        digest.update(str(Path(path).resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def train_fingerprint(cfg: Config, n_train: int, train_paths=None, train_labels=None,
                      val_paths=None, val_labels=None) -> Dict[str, Any]:
    """Fingerprint the training recipe and exact ordered train/validation manifests.

    Paths/labels are optional for backwards compatibility with lightweight callers, but the
    training script supplies them so checkpoint reuse is invalidated when the split changes.
    """
    return {
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "label_smoothing": cfg.label_smoothing,
        "patience": cfg.patience,
        "seed": cfg.seed,
        "val_fraction": cfg.val_fraction,
        "img_size": cfg.img_size,
        "resize": cfg.resize,
        "normalization_mean": list(cfg.mean),
        "normalization_std": list(cfg.std),
        "use_amp": cfg.use_amp,
        "train_transform": {"random_resized_crop_scale": [0.35, 1.0], "horizontal_flip_probability": 0.5},
        "eval_transform": "resize-center-crop-to-tensor-normalize-v1",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "weights": "ResNet50_Weights.IMAGENET1K_V2",
        "n_train": int(n_train),
        "train_manifest_sha256": _split_digest(train_paths, train_labels),
        "val_manifest_sha256": _split_digest(val_paths, val_labels),
    }


def _make_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, train: bool = False,
              use_amp: bool = False) -> Tuple[float, float]:
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False, desc="train" if train else "eval"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def train_model(model: nn.Module, train_loader, val_loader, cfg: Config, device: torch.device,
                ckpt_path: Path, hist_path: Path, fingerprint: Dict[str, Any],
                logger: Optional[logging.Logger] = None) -> Dict[str, list]:
    """AdamW + cosine schedule, best-checkpoint selection (val accuracy, ties broken by val loss),
    early stopping. The complete history is embedded in the checkpoint and also written as JSON."""
    log = logger.info if logger else print
    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = _make_scaler(device, use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, best_loss, bad_epochs = -1.0, float("inf"), 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler, True, use_amp)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, None, None, False, use_amp)
        hist["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        for key, val in zip(["train_loss", "train_acc", "val_loss", "val_acc"], [tr_loss, tr_acc, va_loss, va_acc]):
            hist[key].append(val)
        better = (va_acc > best_acc + 1e-6) or (abs(va_acc - best_acc) <= 1e-6 and va_loss < best_loss)
        if better:
            best_acc, best_loss, bad_epochs = va_acc, va_loss, 0
            save_checkpoint({"model_state": model.state_dict(), "epoch": epoch, "val_acc": va_acc,
                             "val_loss": va_loss, "class_names": CLASS_NAMES, "wnids": WNIDS,
                             "train_cfg": fingerprint, "history": copy.deepcopy(hist),
                             "train_complete": False}, ckpt_path)
        else:
            bad_epochs += 1
        log(f"epoch {epoch:02d}/{cfg.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.0f}s {'(best)' if better else ''}")
        if bad_epochs >= cfg.patience:
            log("Early stopping.")
            break
    # finalise: embed the COMPLETE history in the checkpoint itself (JSON copy kept for convenience)
    ck = load_checkpoint(ckpt_path, map_location="cpu")
    ck["history"], ck["train_complete"] = hist, True
    save_checkpoint(ck, ckpt_path)
    with open(hist_path, "w") as f:
        json.dump(hist, f)
    return hist


def checkpoint_is_reusable(ckpt_path: Path, fingerprint: Dict[str, Any]) -> Tuple[bool, Any]:
    """Reuse only a checkpoint that (a) finished training and (b) was trained with the current config.
    Returns (True, checkpoint) or (False, reason)."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return False, "no checkpoint found"
    ck = load_checkpoint(ckpt_path, map_location="cpu")
    if not ck.get("train_complete", False):
        return False, "existing checkpoint comes from an unfinished/older run"
    if ck.get("train_cfg") != fingerprint:
        return False, "existing checkpoint was trained with a different configuration"
    return True, ck


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses, preds, labels = 0.0, [], []
    for x, y in tqdm(loader, leave=False, desc="evaluate"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses += criterion(logits, y).item() * x.size(0)
        preds.append(logits.argmax(1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {"loss": losses / len(labels), "acc": float((preds == labels).mean()), "preds": preds, "labels": labels}


def load_trained_model(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild the 10-class ResNet50 (no pretrained download), load the best checkpoint, eval mode."""
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
