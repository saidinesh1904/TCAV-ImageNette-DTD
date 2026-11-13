#!/usr/bin/env python
"""Statistical analysis of run_tcav.py results (true CAVs vs random controls, Holm / BH correction)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # run without installing

import pandas as pd

from tcav_imagenette.config import add_common_args, load_config
from tcav_imagenette.statistics import SHOW_COLS, run_statistical_analysis, summarize_results
from tcav_imagenette.utils import ensure_dir, get_logger, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--plots", action="store_true", help="also write the TCAV result figures")
    args = parser.parse_args()

    # config precedence: --config, then <out-dir>/config.json written by the experiment, then defaults
    out_dir = Path(args.out_dir) if args.out_dir else None
    cfg_path = args.config
    if cfg_path is None and out_dir is not None and (out_dir / "config.json").exists():
        cfg_path = out_dir / "config.json"
    cfg = load_config(cfg_path, {"out_dir": args.out_dir, "data_dir": args.data_dir, "pilot": args.pilot})
    out_dir = ensure_dir(cfg.out_path)
    log = get_logger(log_file=out_dir / "analyze_results.log")

    cav_df = pd.read_csv(out_dir / "cav_runs.csv")
    res_df = pd.read_csv(out_dir / "tcav_results_per_run.csv")
    stats_raw, stats_exploratory, stats_confirmatory = run_statistical_analysis(res_df, cfg)

    stats_exploratory.to_csv(out_dir / "stats_exploratory_all_pairs.csv", index=False)
    stats_confirmatory.to_csv(out_dir / "stats_confirmatory_hypotheses.csv", index=False)

    test_acc = None
    metrics_path = out_dir / "test_metrics.json"
    if metrics_path.exists():
        test_acc = load_json(metrics_path).get("test_acc")

    log.info("CONFIRMATORY family (pre-specified hypotheses; Holm within layer x control):\n%s",
             stats_confirmatory[SHOW_COLS].round(4).to_string(index=False))
    log.info("\n%s", summarize_results(cav_df, stats_confirmatory, stats_exploratory, cfg, test_acc))

    if args.plots:
        from tcav_imagenette.visualization import plot_all_tcav_results
        plot_all_tcav_results(cav_df, res_df, stats_confirmatory, cfg, out_dir)


if __name__ == "__main__":
    main()
