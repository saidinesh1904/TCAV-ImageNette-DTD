import hashlib
import io
import tarfile
from pathlib import Path

import pytest
import torch

from tcav_imagenette.config import Config
from tcav_imagenette.data import _valid_imagenette_split, extract_tar
from tcav_imagenette.features import feature_cache_fingerprint, load_features, save_features
from tcav_imagenette.model import train_fingerprint
from tcav_imagenette.utils import sha256_file


def test_training_fingerprint_captures_amp_and_split_manifests(tmp_path):
    paths = [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]
    labels = [0, 1]
    base = Config()
    changed_amp = Config(use_amp=False)
    assert train_fingerprint(base, 2, paths, labels, paths, labels) != train_fingerprint(
        changed_amp, 2, paths, labels, paths, labels
    )
    assert train_fingerprint(base, 2, paths, labels, paths, labels) != train_fingerprint(
        base, 2, paths[::-1], labels[::-1], paths, labels
    )


def test_feature_cache_round_trip_and_fingerprint_validation(tmp_path):
    cfg = Config()
    model_id = "model-checkpoint-sha256"
    cache_id = feature_cache_fingerprint(cfg, model_id)
    concept_feats = {"layer3": {"dotted": torch.randn(3, 1024)}}
    ref_feats = {"layer3": torch.randn(4, 1024)}
    concept_paths = {"dotted": ["a.jpg", "b.jpg", "c.jpg"]}
    ref_paths = ["r1.jpg", "r2.jpg", "r3.jpg", "r4.jpg"]
    path = tmp_path / "features.pt"

    save_features(path, concept_feats, ref_feats, concept_paths, ref_paths, model_id, cache_id)
    loaded = load_features(path, torch.device("cpu"), expected_model_fingerprint=model_id,
                           expected_cache_fingerprint=cache_id)
    assert loaded[2] == concept_paths
    assert loaded[3] == ref_paths
    assert torch.equal(loaded[0]["layer3"]["dotted"], concept_feats["layer3"]["dotted"])
    assert torch.equal(loaded[1]["layer3"], ref_feats["layer3"])

    with pytest.raises(RuntimeError, match="different or unidentified"):
        load_features(path, torch.device("cpu"), expected_model_fingerprint="different-model")
    with pytest.raises(RuntimeError, match="preprocessing/layers"):
        load_features(path, torch.device("cpu"), expected_cache_fingerprint="different-cache")


def test_sha256_file(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"tcav")
    assert sha256_file(path) == hashlib.sha256(b"tcav").hexdigest()


def test_tar_extraction_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar"
    payload = b"not allowed"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises((tarfile.TarError, ValueError, OSError)):
        extract_tar(archive, tmp_path / "extract")
    assert not (tmp_path / "escaped.txt").exists()


def test_tar_extraction_accepts_safe_archive(tmp_path):
    archive = tmp_path / "safe.tar"
    payload = b"safe data"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("dataset/train/example.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    destination = tmp_path / "extract"
    extract_tar(archive, destination)
    assert (destination / "dataset/train/example.txt").read_bytes() == payload


def test_imagenette_split_validation_requires_all_classes_and_images(tmp_path):
    from tcav_imagenette.config import WNIDS

    split = tmp_path / "train"
    split.mkdir()
    for wnid in WNIDS:
        class_dir = split / wnid
        class_dir.mkdir()
        (class_dir / "sample.jpg").write_bytes(b"x")
    assert _valid_imagenette_split(tmp_path, "train")
    (split / WNIDS[0] / "sample.jpg").unlink()
    assert not _valid_imagenette_split(tmp_path, "train")
