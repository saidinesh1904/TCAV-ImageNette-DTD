#!/usr/bin/env python
"""Fine-tune ResNet50 on ImageNette, save the best checkpoint, and evaluate it on val/test."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # run without installing

import pandas as pd

from tcav_imagenette.config import add_common_args, config_from_args, save_config
from tcav_imagenette.data import ensure_imagenette, load_imagenette_splits, make_loaders
from tcav_imagenette.model import (CKPT_NAME, HIST_NAME, build_model, checkpoint_is_reusable, evaluate,
                                   load_trained_model, train_fingerprint, train_model)
from tcav_imagenette.utils import ensure_dir, get_device, get_logger, make_generator, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--force-retrain", action="store_const", const=True, default=None,
                        help="retrain even if a matching completed checkpoint exists")
    parser.add_argument("--no-download", action="store_true", help="do not download ImageNette if missing")
    args = parser.parse_args()

    cfg = config_from_args(args, force_retrain=args.force_retrain)
    out_dir = ensure_dir(cfg.out_path)
    log = get_logger(log_file=out_dir / "train_model.log")
    device = get_device()
    log.info("device: %s | epochs=%d | pilot=%s", device, cfg.epochs, cfg.pilot)

    set_seed(cfg.seed)
    generator = make_generator(cfg.seed)
    root = cfg.data_path / "imagenette2-320"
    if not args.no_download:
        root = ensure_imagenette(cfg)
    splits = load_imagenette_splits(cfg, root)
    log.info("train %d | val %d | test %d", len(splits.train_paths), len(splits.val_paths), len(splits.test_paths))
    train_loader, val_loader, test_loader = make_loaders(cfg, splits, device, generator)

    # one probe batch, as in the notebook (shape sanity check; also advances the shuffling generator
    # exactly like the notebook did before training)
    xb, yb = next(iter(train_loader))
    log.info("batch: %s %s | labels: %s", tuple(xb.shape), xb.dtype, tuple(yb.shape))

    ckpt_path, hist_path = out_dir / CKPT_NAME, out_dir / HIST_NAME
    fingerprint = train_fingerprint(
        cfg, len(splits.train_paths),
        train_paths=splits.train_paths, train_labels=splits.train_labels,
        val_paths=splits.val_paths, val_labels=splits.val_labels,
    )
    reusable, info = checkpoint_is_reusable(ckpt_path, fingerprint)
    if reusable and not cfg.force_retrain:
        log.info("Matching, completed checkpoint found -> skipping training (use --force-retrain to retrain).")
    else:
        log.info("Training from scratch: %s", "force_retrain" if reusable else info)
        set_seed(cfg.seed)
        model = build_model().to(device)
        train_model(model, train_loader, val_loader, cfg, device, ckpt_path, hist_path, fingerprint, log)

    # evaluate the best (by validation accuracy) checkpoint
    model, ckpt = load_trained_model(ckpt_path, device)
    log.info("Loaded best checkpoint from epoch %d (val acc %.4f)", ckpt["epoch"], ckpt["val_acc"])
    val_res, test_res = evaluate(model, val_loader, device), evaluate(model, test_loader, device)
    log.info("val  loss %.4f acc %.4f", val_res["loss"], val_res["acc"])
    log.info("test loss %.4f acc %.4f", test_res["loss"], test_res["acc"])

    pd.DataFrame({"y_true": test_res["labels"], "y_pred": test_res["preds"],
                  "path": splits.test_paths}).to_csv(out_dir / "test_predictions.csv", index=False)
    save_json({"test_acc": test_res["acc"], "test_loss": test_res["loss"],
               "val_acc": val_res["acc"], "val_loss": val_res["loss"]}, out_dir / "test_metrics.json")
    save_config(cfg, out_dir / "config.json")


if __name__ == "__main__":
    main()
