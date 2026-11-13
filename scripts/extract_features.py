#!/usr/bin/env python
"""Extract and save pooled layer3/layer4 features of the DTD concept images and the reference pool."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # run without installing

from tcav_imagenette.config import add_common_args, config_from_args, save_config
from tcav_imagenette.data import build_transforms, load_image_batch, prepare_data
from tcav_imagenette.features import (SplitResNet, check_split, extract_concept_and_reference_features,
                                      feature_cache_fingerprint, save_features)
from tcav_imagenette.model import CKPT_NAME, freeze_model, load_trained_model
from tcav_imagenette.utils import ensure_dir, get_device, get_logger, set_seed, sha256_file

FEATURES_NAME = "features.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--checkpoint", default=None, help=f"model checkpoint (default: <out-dir>/{CKPT_NAME})")
    parser.add_argument("--output", default=None, help=f"feature file (default: <out-dir>/{FEATURES_NAME})")
    parser.add_argument("--no-download", action="store_true", help="do not download datasets if missing")
    args = parser.parse_args()

    cfg = config_from_args(args)
    out_dir = ensure_dir(cfg.out_path)
    log = get_logger(log_file=out_dir / "extract_features.log")
    device = get_device()
    set_seed(cfg.seed)

    data = prepare_data(cfg, download=not args.no_download)
    checkpoint_path = Path(args.checkpoint or out_dir / CKPT_NAME)
    model, ckpt = load_trained_model(checkpoint_path, device)
    model_fingerprint = sha256_file(checkpoint_path)
    cache_fingerprint = feature_cache_fingerprint(cfg, model_fingerprint)
    freeze_model(model)
    split = SplitResNet(model)

    # split network must reproduce the original model and the forward-hook activations
    _, eval_tf = build_transforms(cfg)
    x_chk = load_image_batch(data.splits.test_paths[:8], eval_tf)
    for layer, r in check_split(model, split, x_chk, cfg.tcav_layers, device).items():
        log.info("%s: activation %s -> pooled %s | hook match %s | downstream==model %s (max|diff| %.2e)",
                 layer, r["activation_shape"], r["pooled_shape"], r["matches_hook"],
                 r["downstream_matches_model"], r["max_abs_diff"])
        if not (r["matches_hook"] and r["downstream_matches_model"]):
            raise RuntimeError(f"SplitResNet is not equivalent to the original model at {layer}")

    concept_feats, ref_feats = extract_concept_and_reference_features(
        split, data.concept_paths, data.ref_paths, cfg, device)
    for layer in cfg.tcav_layers:
        log.info("%s | concept feats: %s | reference: %s", layer,
                 {c: tuple(v.shape) for c, v in concept_feats[layer].items()}, tuple(ref_feats[layer].shape))

    out_path = Path(args.output or out_dir / FEATURES_NAME)
    save_features(out_path, concept_feats, ref_feats, data.concept_paths, data.ref_paths,
                  model_fingerprint, cache_fingerprint)
    save_config(cfg, out_dir / "config.json")
    log.info("saved features to %s", out_path)


if __name__ == "__main__":
    main()
