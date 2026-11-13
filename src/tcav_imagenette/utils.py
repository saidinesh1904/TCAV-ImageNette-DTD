"""Seeding, device selection, logging, directories, JSON and checkpoint utilities."""
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

PathLike = Union[str, Path]


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch and make cuDNN deterministic (as in the notebook)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive numpy/python seeds from the torch worker seed."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_generator(seed: int) -> torch.Generator:
    """Generator used for DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device(prefer: Optional[str] = None) -> torch.device:
    """CUDA if available (or the explicitly requested device), otherwise CPU."""
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str = "tcav_imagenette", level: int = logging.INFO,
               log_file: Optional[PathLike] = None) -> logging.Logger:
    """Console (and optional file) logger; safe to call repeatedly."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(str(log_file))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: PathLike) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)



def sha256_file(path: PathLike, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def save_checkpoint(state: Dict[str, Any], path: PathLike) -> None:
    torch.save(state, str(path))


def load_checkpoint(path: PathLike, map_location: Any = "cpu") -> Dict[str, Any]:
    return torch.load(str(path), map_location=map_location)
