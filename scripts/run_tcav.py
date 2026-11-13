#!/usr/bin/env python
"""Run the TCAV experiments: class gradients, derivative checks, true-concept CAVs and random controls."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # run without installing

from tcav_imagenette.cav import fit_cav_run
from tcav_imagenette.config import CLASS_NAMES, add_common_args, config_from_args, save_config
from tcav_imagenette.controls import run_experiments
from tcav_imagenette.data import build_transforms, load_image_batch, prepare_data
from tcav_imagenette.features import (SplitResNet, extract_concept_and_reference_features,
                                      feature_cache_fingerprint, load_features, save_features)
from tcav_imagenette.model import CKPT_NAME, freeze_model, load_trained_model
from tcav_imagenette.tcav import check_gradients, compute_all_gradients, verify_directional_derivative
from tcav_imagenette.utils import ensure_dir, get_device, get_logger, save_json, set_seed, sha256_file

FEATURES_NAME = "features.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--checkpoint", default=None, help=f"model checkpoint (default: <out-dir>/{CKPT_NAME})")
    parser.add_argument("--features", default=None,
                        help=f"feature file from extract_features.py (default: <out-dir>/{FEATURES_NAME}; "
                             "extracted on the fly if it does not exist)")
    parser.add_argument("--only-correct", action="store_const", const=True, default=None,
                        help="X_k = only correctly classified class-k test images (robustness check)")
    parser.add_argument("--no-download", action="store_true", help="do not download datasets if missing")
    args = parser.parse_args()

    cfg = config_from_args(args, tcav_only_correct=args.only_correct)
    out_dir = ensure_dir(cfg.out_path)
    log = get_logger(log_file=out_dir / "run_tcav.log")
    device = get_device()
    set_seed(cfg.seed)
    log.info("device: %s | layers=%s | primary=%s | cav_runs=%d | random_runs=%d | pilot=%s",
             device, cfg.tcav_layers, cfg.primary_layer, cfg.n_cav_runs, cfg.n_random_runs, cfg.pilot)

    data = prepare_data(cfg, download=not args.no_download)
    checkpoint_path = Path(args.checkpoint or out_dir / CKPT_NAME)
    model, _ = load_trained_model(checkpoint_path, device)
    model_fingerprint = sha256_file(checkpoint_path)
    cache_fingerprint = feature_cache_fingerprint(cfg, model_fingerprint)
    freeze_model(model)
    split = SplitResNet(model)
    test_paths_by_class = data.splits.test_paths_by_class

    # ---- concept / reference features (reuse the saved file when it matches the current selection) ----
    feat_path = Path(args.features or out_dir / FEATURES_NAME)
    if feat_path.exists():
        concept_feats, ref_feats, concept_paths, ref_paths = load_features(
            feat_path, device,
            expected_model_fingerprint=model_fingerprint,
            expected_cache_fingerprint=cache_fingerprint,
        )
        if concept_paths != data.concept_paths or ref_paths != data.ref_paths:
            raise RuntimeError(f"{feat_path} was extracted for a different concept/reference selection; "
                               "re-run extract_features.py")
        log.info("loaded features from %s", feat_path)
    else:
        concept_feats, ref_feats = extract_concept_and_reference_features(
            split, data.concept_paths, data.ref_paths, cfg, device)
        save_features(feat_path, concept_feats, ref_feats, data.concept_paths, data.ref_paths,
                      model_fingerprint, cache_fingerprint)
        log.info("extracted and saved features to %s", feat_path)

    # ---- directional-derivative ingredients: grad of logit h_k w.r.t. pooled feature, per image ----
    grads, test_pred, correct_mask = compute_all_gradients(split, test_paths_by_class, cfg, device)
    rel_std = check_gradients(grads, test_paths_by_class, cfg.tcav_layers)
    for layer, rel in rel_std.items():
        log.info("%s: mean relative std of gradient across images = %.4f%s", layer, rel,
                 "  (image-independent gradient: linear head -> each CAV gives TCAV in {0,1})" if rel < 1e-3 else "")

    # ---- verification of the directional derivative on the first hypothesis pair, every layer ----
    _, eval_tf = build_transforms(cfg)
    c0 = list(cfg.concept_class_hypotheses.keys())[0]
    k0 = CLASS_NAMES.index(cfg.concept_class_hypotheses[c0])
    x_chk = load_image_batch(test_paths_by_class[k0][:16], eval_tf).to(device)
    checks = {}
    for layer in cfg.tcav_layers:
        cav = None
        for check_run in range(cfg.n_cav_runs):
            candidate = fit_cav_run(layer, c0, cfg.concepts.index(c0), check_run, "concept",
                                    concept_feats, ref_feats, cfg)
            if candidate["valid"]:
                cav = candidate
                break
        if cav is None:
            raise RuntimeError(f"No valid CAV found for derivative verification: {c0}/{layer}; "
                               "inspect CAV validation accuracy and concept features")
        checks[layer] = verify_directional_derivative(split, x_chk, layer, k0, cav["v"], grads[layer][k0])
        log.info("[%s, class '%s'] %s", layer, CLASS_NAMES[k0], checks[layer])
    save_json(checks, out_dir / "directional_derivative_check.json")

    # ---- repeated experiments: true CAVs + shuffled-label and random-reference controls ----
    set_seed(cfg.seed)
    cav_df, res_df = run_experiments(cfg, concept_feats, ref_feats, grads, correct_mask)
    log.info("CAV runs: %d | TCAV result rows: %d", len(cav_df), len(res_df))
    concept_runs = cav_df[cav_df.kind == "concept"]
    log.info("concept CAVs valid: %d/%d", int(concept_runs.valid.sum()), len(concept_runs))

    cav_df.to_csv(out_dir / "cav_runs.csv", index=False)
    res_df.to_csv(out_dir / "tcav_results_per_run.csv", index=False)
    save_config(cfg, out_dir / "config.json")
    log.info("saved results to %s", out_dir)


if __name__ == "__main__":
    main()
