"""Statistical analysis: true-concept CAVs vs random controls.

For each (layer, concept, class) pair the per-run TCAV scores of the true concept CAVs are compared
with those of each random control (same layer; for ``shuffled_labels`` the same concept):

* primary test: two-sided Mann-Whitney U on the TCAV scores;
* secondary: Welch's t-test on TCAV scores (NaN when both groups have zero variance -> treated as p=1
  for correction) and Mann-Whitney U on the mean directional derivative;
* effect sizes: difference in mean TCAV and Cliff's delta;
* multiple-comparison correction (Holm and Benjamini-Hochberg) within each (layer x control) family:
  confirmatory = only the pre-specified hypothesis pairs, exploratory = all concept x class pairs.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import CLASS_NAMES, NUM_CLASSES, RANDOM_REF_NAME, Config

P_COLS = ["p_mwu_tcav", "p_welch_tcav", "p_mwu_meanS"]

SHOW_COLS = ["layer", "concept", "class_name", "control", "n_actual", "mean_actual", "std_actual",
             "mean_control", "std_control", "diff_mean", "cliffs_delta", "p_mwu_tcav", "p_mwu_tcav_holm",
             "p_welch_tcav_holm", "p_mwu_meanS_holm", "supported_holm", "direction"]


def cliffs_delta(a, b) -> float:
    """P(a > b) - P(a < b) over all pairs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((a[:, None] > b[None, :]).mean() - (a[:, None] < b[None, :]).mean())


def safe_mwu(a, b) -> float:
    """Two-sided Mann-Whitney U p-value (NaN if a group has < 2 values; 1.0 if the test is undefined)."""
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    except ValueError:
        p = 1.0
    return 1.0 if np.isnan(p) else float(p)


def safe_welch(a, b) -> float:
    """Welch's t-test p-value (NaN if a group has < 2 values or both groups have zero variance)."""
    if len(a) < 2 or len(b) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def compute_raw_stats(res_df: pd.DataFrame, cfg: Config, class_names: Sequence[str] = CLASS_NAMES) -> pd.DataFrame:
    """Uncorrected comparisons of true concept CAVs vs each control, for every (layer, concept, class)."""
    groups = {key: g for key, g in res_df.groupby(["layer", "kind", "concept", "class_idx"])}
    hyp_pairs = {(c, list(class_names).index(kname)) for c, kname in cfg.concept_class_hypotheses.items()}

    rows = []
    for layer in cfg.tcav_layers:
        for concept in cfg.concepts:
            for k in range(NUM_CLASSES):
                act = groups.get((layer, "concept", concept, k))
                if act is None:
                    continue
                for control in cfg.controls:
                    ctrl_concept = concept if control == "shuffled_labels" else RANDOM_REF_NAME
                    ctl = groups.get((layer, control, ctrl_concept, k))
                    if ctl is None:
                        continue
                    a, r = act["tcav"].values, ctl["tcav"].values
                    rows.append({
                        "layer": layer, "concept": concept, "class_idx": k, "class_name": class_names[k],
                        "control": control, "is_hypothesis": (concept, k) in hyp_pairs,
                        "n_actual": len(a), "n_control": len(r),
                        "mean_actual": a.mean(), "std_actual": a.std(ddof=1) if len(a) > 1 else np.nan,
                        "mean_control": r.mean(), "std_control": r.std(ddof=1) if len(r) > 1 else np.nan,
                        "diff_mean": a.mean() - r.mean(), "cliffs_delta": cliffs_delta(a, r),
                        "meanS_actual": act["mean_S"].mean(), "meanS_control": ctl["mean_S"].mean(),
                        "p_mwu_tcav": safe_mwu(a, r), "p_welch_tcav": safe_welch(a, r),
                        "p_mwu_meanS": safe_mwu(act["mean_S"].values, ctl["mean_S"].values)})
    return pd.DataFrame(rows)


def add_corrections(df: pd.DataFrame, alpha: float, family_cols: Tuple[str, ...] = ("layer", "control")) -> pd.DataFrame:
    """Holm and Benjamini-Hochberg adjusted p-values within each family (default: layer x control).
    ``supported_holm`` = Holm-adjusted Mann-Whitney p on TCAV scores < alpha."""
    df = df.copy().reset_index(drop=True)
    for pc in P_COLS:
        df[pc + "_holm"] = np.nan
        df[pc + "_bh"] = np.nan
    if df.empty:
        df["supported_holm"] = pd.Series(dtype=bool)
        df["direction"] = pd.Series(dtype=object)
        return df
    for _, idx in df.groupby(list(family_cols)).groups.items():
        for pc in P_COLS:
            p = df.loc[idx, pc].fillna(1.0).values
            df.loc[idx, pc + "_holm"] = multipletests(p, alpha=alpha, method="holm")[1]
            df.loc[idx, pc + "_bh"] = multipletests(p, alpha=alpha, method="fdr_bh")[1]
    df["supported_holm"] = df["p_mwu_tcav_holm"] < alpha
    df["direction"] = np.where(df["diff_mean"] > 0, "above control",
                               np.where(df["diff_mean"] < 0, "below control", "equal"))
    return df


def run_statistical_analysis(res_df: pd.DataFrame, cfg: Config, class_names: Sequence[str] = CLASS_NAMES
                             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (stats_raw, stats_exploratory, stats_confirmatory).

    * exploratory: all concept x class pairs per (layer, control) family;
    * confirmatory: only the pre-specified hypothesis pairs (``cfg.concept_class_hypotheses``).
    Only Holm-adjusted p < alpha in the confirmatory family counts as "supported".
    """
    stats_raw = compute_raw_stats(res_df, cfg, class_names)
    if stats_raw.empty:
        return stats_raw, add_corrections(stats_raw, cfg.alpha), add_corrections(stats_raw, cfg.alpha)
    stats_exploratory = add_corrections(stats_raw, cfg.alpha)
    stats_confirmatory = add_corrections(stats_raw[stats_raw.is_hypothesis], cfg.alpha)
    return stats_raw, stats_exploratory, stats_confirmatory


def summarize_results(cav_df: pd.DataFrame, stats_confirmatory: pd.DataFrame, stats_exploratory: pd.DataFrame,
                      cfg: Config, test_acc: Optional[float] = None) -> str:
    """Text summary computed from the results (no hard-coded conclusions)."""
    lines = []
    if test_acc is not None:
        lines.append(f"Test accuracy of fine-tuned ResNet50: {test_acc:.4f}\n")
    for layer in cfg.tcav_layers:
        cd = cav_df[(cav_df.layer == layer) & (cav_df.kind == "concept")]
        lines.append(f"=== {layer}: {int(cd.valid.sum())}/{len(cd)} concept CAVs passed the validity checks "
                     f"(mean val acc {cd.val_acc.mean():.3f}) ===")
        for control in cfg.controls:
            s = stats_confirmatory[(stats_confirmatory.layer == layer) & (stats_confirmatory.control == control)]
            for _, r in s.iterrows():
                verdict = "SUPPORTED" if r.supported_holm else "not supported"
                lines.append(f"  [{control:16s}] {r.concept:>9s} -> {r.class_name:<16s} "
                             f"TCAV {r.mean_actual:.3f}+-{r.std_actual:.3f} "
                             f"vs control {r.mean_control:.3f}+-{r.std_control:.3f} | "
                             f"Holm p={r.p_mwu_tcav_holm:.3g} | {verdict} ({r.direction})")
        ex = stats_exploratory[(stats_exploratory.layer == layer) & (stats_exploratory.control == "shuffled_labels")]
        lines.append(f"  exploratory: {int(ex.supported_holm.sum())}/{len(ex)} concept-class pairs have "
                     f"Holm p<{cfg.alpha} vs shuffled-label control (exploratory, not confirmatory)\n")
    return "\n".join(lines)
