"""Experiment configuration.

Mirrors the ``CFG`` class of the executed notebook (``c1.ipynb``). Values below are the
notebook's final-run values; ``configs/default.json`` holds the same values.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# --------------------------------------------------------------------------- constants
IMAGENETTE_CLASSES = OrderedDict([
    ("n01440764", "tench"), ("n02102040", "English springer"), ("n02979186", "cassette player"),
    ("n03000684", "chain saw"), ("n03028079", "church"), ("n03394916", "French horn"),
    ("n03417042", "garbage truck"), ("n03425413", "gas pump"), ("n03445777", "golf ball"),
    ("n03888257", "parachute"),
])
WNIDS = list(IMAGENETTE_CLASSES.keys())
CLASS_NAMES = list(IMAGENETTE_CLASSES.values())
NUM_CLASSES = len(CLASS_NAMES)

LAYER_ORDER = ["layer1", "layer2", "layer3", "layer4"]
LAYER_DIMS = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}

# kind of CAV run -> integer used in the per-run RNG seed
KIND_ID = {"concept": 0, "shuffled_labels": 1, "random_reference": 2}
# placeholder "concept" name of the concept-independent random-reference control
RANDOM_REF_NAME = "__random_reference__"
# RNG-seed index used for the random-reference control (it has no concept index)
RANDOM_REF_INDEX = 100

REFERENCE_SOURCES = ("dtd_other", "imagenette_train")


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    # ---- reproducibility ----
    seed: int = 42

    # ---- paths (relative paths are resolved against the current working directory) ----
    data_dir: str = "tcav_data"
    out_dir: str = "tcav_outputs"

    # ---- datasets ----
    imagenette_url: str = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    dtd_url: str = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    val_fraction: float = 0.10          # held out from the official ImageNette *train* folder
    # (the official ImageNette *val* folder is the TEST set)

    # ---- preprocessing (ImageNet statistics) ----
    resize: int = 256
    img_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ---- fine-tuning ----
    batch_size: int = 64
    num_workers: int = 2
    epochs: int = 6
    lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    patience: int = 3
    use_amp: bool = True                # mixed precision for fine-tuning only (TCAV uses fp32)
    force_retrain: bool = False
    pilot: bool = False                 # True -> overrides epochs / run counts with the pilot_* values

    # ---- concepts (DTD categories) and pre-specified hypotheses ----
    concepts: List[str] = field(default_factory=lambda: ["dotted", "striped", "banded", "zigzagged", "woven"])
    concept_class_hypotheses: Dict[str, str] = field(default_factory=lambda: {
        "dotted": "golf ball",
        "striped": "English springer",
        "banded": "parachute",
        "zigzagged": "chain saw",
        "woven": "cassette player",
    })
    n_concept_imgs: int = 120
    reference_source: str = "dtd_other"     # "dtd_other" or "imagenette_train"
    ref_pool_size: int = 1500
    extra_exclude: List[str] = field(default_factory=lambda: ["polka-dotted", "lined", "stratified", "crosshatched"])

    # ---- TCAV ----
    tcav_layers: List[str] = field(default_factory=lambda: ["layer4", "layer3"])
    primary_layer: str = "layer3"
    tcav_batch_size: int = 64
    tcav_only_correct: bool = False     # X_k = ALL ground-truth class-k test images
    cav_val_frac: float = 0.20
    cav_steps: int = 200
    cav_lr: float = 0.02
    cav_l2: float = 1e-2
    cav_min_acc: float = 0.70           # concept CAVs below this validation accuracy are discarded
    n_cav_runs: int = 30                # repeated CAVs per (layer, concept)
    n_random_runs: int = 50             # repeated random CAVs per control
    controls: List[str] = field(default_factory=lambda: ["shuffled_labels", "random_reference"])
    alpha: float = 0.05

    # ---- pilot-run overrides (used only when pilot=True) ----
    pilot_epochs: int = 2
    pilot_cav_runs: int = 5
    pilot_random_runs: int = 10

    def __post_init__(self) -> None:
        self.mean = tuple(self.mean)
        self.std = tuple(self.std)

    # ---- helpers ----
    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def out_path(self) -> Path:
        return Path(self.out_dir)

    def apply_pilot(self) -> "Config":
        """Apply the pilot overrides (EPOCHS, N_CAV_RUNS, N_RANDOM_RUNS) in place."""
        self.epochs = self.pilot_epochs
        self.n_cav_runs = self.pilot_cav_runs
        self.n_random_runs = self.pilot_random_runs
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mean"], d["std"] = list(self.mean), list(self.std)
        return d


def validate_config(cfg: Config) -> Config:
    """Raise ``ValueError`` if the configuration is inconsistent. Returns ``cfg``."""
    errs: List[str] = []
    for name in ("tcav_layers",):
        bad = [l for l in getattr(cfg, name) if l not in LAYER_DIMS]
        if bad:
            errs.append(f"{name} contains unknown layers {bad}; valid: {list(LAYER_DIMS)}")
    if cfg.primary_layer not in cfg.tcav_layers:
        errs.append(f"primary_layer '{cfg.primary_layer}' must be one of tcav_layers {cfg.tcav_layers}")
    if len(set(cfg.concepts)) != len(cfg.concepts):
        errs.append("concepts contains duplicates")
    for c, k in cfg.concept_class_hypotheses.items():
        if c not in cfg.concepts:
            errs.append(f"hypothesis concept '{c}' is not in concepts")
        if k not in CLASS_NAMES:
            errs.append(f"hypothesis target class '{k}' is not an ImageNette class")
    bad_ctrl = [c for c in cfg.controls if c not in ("shuffled_labels", "random_reference")]
    if bad_ctrl:
        errs.append(f"unknown controls {bad_ctrl}")
    if cfg.reference_source not in REFERENCE_SOURCES:
        errs.append(f"reference_source must be one of {REFERENCE_SOURCES}")
    if len(cfg.mean) != 3 or len(cfg.std) != 3:
        errs.append("mean/std must have 3 entries")
    for name in ("val_fraction", "cav_val_frac", "alpha", "cav_min_acc"):
        if not 0.0 < getattr(cfg, name) < 1.0:
            errs.append(f"{name} must be in (0, 1)")
    for name in ("epochs", "batch_size", "tcav_batch_size", "cav_steps", "n_concept_imgs",
                 "n_cav_runs", "n_random_runs", "patience", "img_size", "resize"):
        if getattr(cfg, name) < 1:
            errs.append(f"{name} must be >= 1")
    if cfg.ref_pool_size < 2 * cfg.n_concept_imgs:
        errs.append("ref_pool_size must be at least 2 * n_concept_imgs (random_reference control)")
    if cfg.lr <= 0 or cfg.cav_lr <= 0:
        errs.append("learning rates must be positive")
    if errs:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errs))
    return cfg


def load_config(path: Optional[Union[str, Path]] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Build a validated ``Config`` from defaults, an optional JSON file, and optional overrides.

    ``None`` values in ``overrides`` are ignored (convenient for argparse). If ``pilot`` is true the
    pilot values replace ``epochs``, ``n_cav_runs`` and ``n_random_runs``.
    """
    values: Dict[str, Any] = {}
    if path is not None:
        with open(path, "r") as f:
            values.update(json.load(f))
    if overrides:
        values.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    cfg = Config(**values)
    if cfg.pilot:
        cfg.apply_pilot()
    return validate_config(cfg)


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    with open(path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)


# --------------------------------------------------------------------------- CLI helpers
def add_common_args(parser) -> None:
    """Arguments shared by all scripts."""
    parser.add_argument("--config", default=None, help="path to a JSON config (defaults: built-in values)")
    parser.add_argument("--data-dir", default=None, help="dataset directory (overrides config)")
    parser.add_argument("--out-dir", default=None, help="output directory (overrides config)")
    parser.add_argument("--pilot", action="store_const", const=True, default=None,
                        help="pilot run: use the small pilot_* epoch / run-count settings")


def config_from_args(args, **extra) -> Config:
    """Build the validated config from the common CLI arguments (+ extra overrides)."""
    overrides = {"data_dir": args.data_dir, "out_dir": args.out_dir, "pilot": args.pilot}
    overrides.update(extra)
    return load_config(args.config, overrides)
