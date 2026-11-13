"""Dataset download/loading, ImageNette splits, DTD concept selection, reference pool, DataLoaders."""
from __future__ import annotations

import tarfile
from pathlib import PurePosixPath, PureWindowsPath
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from .config import CLASS_NAMES, NUM_CLASSES, WNIDS, Config
from .utils import seed_worker


# --------------------------------------------------------------------------- download helpers
def download_file(url: str, dest: Path) -> Path:
    """Stream ``url`` to ``dest`` (skipped if it already exists and is non-empty)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.rename(dest)
    return dest


def extract_tar(archive: Path, out_dir: Path) -> None:
    """Extract a dataset archive without allowing path traversal or archive links.

    Python versions supporting the standard ``data`` extraction filter use it directly.
    Older versions receive an explicit member validation before extraction.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(path=root, filter="data")
        except TypeError:  # Python versions whose TarFile.extractall has no filter argument
            members = tar.getmembers()
            for member in members:
                posix_name = PurePosixPath(member.name)
                windows_name = PureWindowsPath(member.name)
                if posix_name.is_absolute() or windows_name.is_absolute() or windows_name.drive:
                    raise ValueError(f"unsafe absolute path in archive: {member.name!r}")
                if ".." in posix_name.parts or ".." in windows_name.parts:
                    raise ValueError(f"unsafe parent traversal in archive: {member.name!r}")
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise ValueError(f"unsupported link/device entry in archive: {member.name!r}")
                target = (root / member.name).resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"archive member escapes extraction directory: {member.name!r}")
            tar.extractall(path=root, members=members)


def find_dtd_images_dir(search_roots: Sequence[Path]) -> Optional[Path]:
    """The DTD ``images`` folder is the one containing 47 category sub-folders."""
    for root in search_roots:
        if not Path(root).exists():
            continue
        for cand in Path(root).rglob("images"):
            if cand.is_dir() and sum(1 for d in cand.iterdir() if d.is_dir()) == 47:
                return cand
    return None


def _valid_imagenette_split(root: Path, split: str) -> bool:
    """Check that a split contains all expected class folders and at least one image per class."""
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        return False
    for wnid in WNIDS:
        class_dir = split_dir / wnid
        if not class_dir.is_dir() or not any(p.is_file() for p in class_dir.iterdir()):
            return False
    return True


def ensure_imagenette(cfg: Config) -> Path:
    """Return a complete ImageNette root, downloading/extracting it if either split is incomplete."""
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    root = cfg.data_path / "imagenette2-320"
    complete = _valid_imagenette_split(root, "train") and _valid_imagenette_split(root, "val")
    if not complete:
        arc = download_file(cfg.imagenette_url, cfg.data_path / "imagenette2-320.tgz")
        extract_tar(arc, cfg.data_path)
    if not (_valid_imagenette_split(root, "train") and _valid_imagenette_split(root, "val")):
        raise FileNotFoundError(
            f"ImageNette extraction is incomplete under {root}; expected train/val folders "
            "with all ten WNID class folders containing images. Remove the partial extraction and retry."
        )
    return root


def ensure_dtd(cfg: Config) -> Path:
    """Return the DTD ``images`` directory, downloading it (with torchvision fallback) if needed."""
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    roots = [cfg.data_path / "dtd", cfg.data_path / "tv_dtd"]
    images_dir = find_dtd_images_dir(roots)
    if images_dir is None:
        try:
            arc = download_file(cfg.dtd_url, cfg.data_path / "dtd-r1.0.1.tar.gz")
            extract_tar(arc, cfg.data_path)
        except Exception:
            datasets.DTD(root=str(cfg.data_path / "tv_dtd"), split="train", download=True)
        images_dir = find_dtd_images_dir(roots)
    if images_dir is None:
        raise FileNotFoundError("DTD images folder not found - check the download.")
    return images_dir


# --------------------------------------------------------------------------- preprocessing
def build_transforms(cfg: Config) -> Tuple[transforms.Compose, transforms.Compose]:
    """(train_tf, eval_tf). ``eval_tf`` is the single preprocessing used for validation/test,
    DTD concept images, reference images and TCAV test images."""
    normalize = transforms.Normalize(cfg.mean, cfg.std)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize])
    eval_tf = transforms.Compose([
        transforms.Resize(cfg.resize),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(), normalize])
    return train_tf, eval_tf


class ImagePathDataset(Dataset):
    """Loads images from file paths (always converted to RGB). ``labels=None`` -> label -1."""

    def __init__(self, paths, labels=None, transform=None):
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = -1 if self.labels is None else int(self.labels[i])
        return img, y


def make_loader(ds: Dataset, batch_size: int, cfg: Config, device: torch.device,
                shuffle: bool = False, train: bool = False,
                generator: Optional[torch.Generator] = None) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=train,
                      num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
                      worker_init_fn=seed_worker, generator=generator if shuffle else None)


def load_image_batch(paths: Sequence[str], transform) -> torch.Tensor:
    """Stack preprocessed images of ``paths`` into a (N, 3, H, W) tensor."""
    return torch.stack([transform(Image.open(p).convert("RGB")) for p in paths])


# --------------------------------------------------------------------------- ImageNette
@dataclass
class ImageNetteSplits:
    root: Path
    trainval_paths: np.ndarray
    trainval_labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_paths: np.ndarray
    train_labels: np.ndarray
    val_paths: np.ndarray
    val_labels: np.ndarray
    test_paths: np.ndarray
    test_labels: np.ndarray
    test_paths_by_class: Dict[int, List[str]]


def load_imagenette_splits(cfg: Config, root: Path) -> ImageNetteSplits:
    """Stratified train/val split of the official ``train`` folder; official ``val`` folder = test set."""
    train_folder = datasets.ImageFolder(Path(root) / "train")     # used only for file paths / labels
    test_folder = datasets.ImageFolder(Path(root) / "val")
    if train_folder.classes != WNIDS or test_folder.classes != WNIDS:
        raise ValueError("unexpected ImageNette class folders")

    trainval_paths = np.array([p for p, _ in train_folder.samples])
    trainval_labels = np.array([l for _, l in train_folder.samples])
    train_idx, val_idx = train_test_split(
        np.arange(len(trainval_paths)), test_size=cfg.val_fraction,
        stratify=trainval_labels, random_state=cfg.seed)

    train_paths, train_labels = trainval_paths[train_idx], trainval_labels[train_idx]
    val_paths, val_labels = trainval_paths[val_idx], trainval_labels[val_idx]
    test_paths = np.array([p for p, _ in test_folder.samples])
    test_labels = np.array([l for _, l in test_folder.samples])

    # leakage checks
    assert set(train_paths).isdisjoint(val_paths)
    assert set(train_paths).isdisjoint(test_paths) and set(val_paths).isdisjoint(test_paths)

    test_paths_by_class = {k: [p for p, l in zip(test_paths, test_labels) if l == k] for k in range(NUM_CLASSES)}
    return ImageNetteSplits(Path(root), trainval_paths, trainval_labels, train_idx, val_idx,
                            train_paths, train_labels, val_paths, val_labels,
                            test_paths, test_labels, test_paths_by_class)


def make_loaders(cfg: Config, splits: ImageNetteSplits, device: torch.device,
                 generator: torch.Generator):
    """(train_loader, val_loader, test_loader) for fine-tuning / evaluation."""
    train_tf, eval_tf = build_transforms(cfg)
    train_loader = make_loader(ImagePathDataset(splits.train_paths, splits.train_labels, train_tf),
                               cfg.batch_size, cfg, device, shuffle=True, train=True, generator=generator)
    val_loader = make_loader(ImagePathDataset(splits.val_paths, splits.val_labels, eval_tf),
                             cfg.batch_size, cfg, device)
    test_loader = make_loader(ImagePathDataset(splits.test_paths, splits.test_labels, eval_tf),
                              cfg.batch_size, cfg, device)
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- DTD
def load_dtd(images_dir: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """Category names are read from the downloaded folders (not typed by hand)."""
    images_dir = Path(images_dir)
    categories = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    paths = {c: sorted(str(p) for p in (images_dir / c).glob("*.jpg")) for c in categories}
    if len(categories) != 47:
        raise ValueError(f"expected 47 DTD categories, found {len(categories)}")
    return categories, paths


def validate_concepts(cfg: Config, dtd_categories: Sequence[str]) -> None:
    missing = [c for c in cfg.concepts if c not in dtd_categories]
    if missing:
        raise ValueError(f"concepts not in DTD: {missing}")
    if not all(k in CLASS_NAMES for k in cfg.concept_class_hypotheses.values()):
        raise ValueError("unknown target class in hypotheses")
    if not all(c in cfg.concepts for c in cfg.concept_class_hypotheses):
        raise ValueError("hypothesis concept not in concepts")
    unknown_excl = [e for e in cfg.extra_exclude if e not in dtd_categories]
    if unknown_excl:
        raise ValueError(f"extra_exclude contains non-DTD names: {unknown_excl}")


def select_concept_and_reference_paths(cfg: Config, dtd_categories: Sequence[str],
                                       dtd_paths: Dict[str, List[str]],
                                       splits: Optional[ImageNetteSplits] = None
                                       ) -> Tuple[Dict[str, List[str]], List[str]]:
    """Concept image paths (``n_concept_imgs`` per concept) and the reference pool.

    The reference pool is drawn from the other DTD categories (concept categories and
    ``extra_exclude`` removed) or from the ImageNette *training* portion only (never test images).
    The RNG draw order (concepts in order, then reference pool) matches the notebook.
    """
    rng_sel = np.random.default_rng(cfg.seed)
    concept_paths: Dict[str, List[str]] = {}
    for c in cfg.concepts:
        if len(dtd_paths[c]) < cfg.n_concept_imgs:
            raise ValueError(f"{c}: only {len(dtd_paths[c])} images")
        concept_paths[c] = [str(p) for p in rng_sel.choice(dtd_paths[c], size=cfg.n_concept_imgs, replace=False)]

    excluded = set(cfg.concepts) | set(cfg.extra_exclude)
    if cfg.reference_source == "dtd_other":
        ref_pool = [p for c in dtd_categories if c not in excluded for p in dtd_paths[c]]
    elif cfg.reference_source == "imagenette_train":
        if splits is None:
            raise ValueError("ImageNette splits are required for reference_source='imagenette_train'")
        ref_pool = [str(splits.trainval_paths[i]) for i in splits.train_idx]   # no test leakage
    else:
        raise ValueError(cfg.reference_source)

    ref_paths = [str(p) for p in rng_sel.choice(ref_pool, size=min(cfg.ref_pool_size, len(ref_pool)), replace=False)]
    if len(ref_paths) < 2 * cfg.n_concept_imgs:
        raise ValueError("reference pool too small for the random_reference control")
    return concept_paths, ref_paths


@dataclass
class DataBundle:
    splits: ImageNetteSplits
    dtd_images_dir: Path
    dtd_categories: List[str]
    dtd_paths: Dict[str, List[str]]
    concept_paths: Dict[str, List[str]]
    ref_paths: List[str]


def prepare_data(cfg: Config, download: bool = True) -> DataBundle:
    """Locate (optionally download) both datasets and build splits, concept images and reference pool."""
    if download:
        imagenette_root = ensure_imagenette(cfg)
        dtd_images_dir = ensure_dtd(cfg)
    else:
        imagenette_root = cfg.data_path / "imagenette2-320"
        dtd_images_dir = find_dtd_images_dir([cfg.data_path / "dtd", cfg.data_path / "tv_dtd"])
        if not (imagenette_root / "train").exists() or dtd_images_dir is None:
            raise FileNotFoundError(f"datasets not found under {cfg.data_path} (run without --no-download)")
    splits = load_imagenette_splits(cfg, imagenette_root)
    dtd_categories, dtd_paths = load_dtd(dtd_images_dir)
    validate_concepts(cfg, dtd_categories)
    concept_paths, ref_paths = select_concept_and_reference_paths(cfg, dtd_categories, dtd_paths, splits)
    return DataBundle(splits, dtd_images_dir, dtd_categories, dtd_paths, concept_paths, ref_paths)
