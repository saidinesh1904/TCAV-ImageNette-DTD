"""Random-CAV controls and the repeated experiment loop.

Both controls are built WITHOUT the true concept labels and are evaluated with the identical
pipeline as true concept CAVs (same classifier, split, normalisation, TCAV computation):

* ``shuffled_labels``  : the concept dataset's features with a random balanced permutation of the labels
  (per concept) - keeps the feature pool and class balance, destroys the concept signal.
* ``random_reference`` : a random subset of the reference pool vs. another random subset (the control
  of the original TCAV paper); concept-independent, so it is run once per layer and shared by all concepts.
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from tqdm.auto import tqdm

from .cav import fit_cav_run
from .config import CLASS_NAMES, NUM_CLASSES, RANDOM_REF_INDEX, RANDOM_REF_NAME, Config
from .tcav import tcav_for_cav

Job = Tuple[str, str, int, str, int]      # (layer, concept, concept_idx, kind, n_runs)


def build_jobs(cfg: Config) -> List[Job]:
    """Experiment plan: for every layer, ``n_cav_runs`` true CAVs per concept plus the requested controls."""
    jobs: List[Job] = []
    for layer in cfg.tcav_layers:
        for ci, concept in enumerate(cfg.concepts):
            jobs.append((layer, concept, ci, "concept", cfg.n_cav_runs))
            if "shuffled_labels" in cfg.controls:
                jobs.append((layer, concept, ci, "shuffled_labels", cfg.n_random_runs))
        if "random_reference" in cfg.controls:
            jobs.append((layer, RANDOM_REF_NAME, RANDOM_REF_INDEX, "random_reference", cfg.n_random_runs))
    return jobs


def run_experiments(cfg: Config, concept_feats, ref_feats, grads, correct_mask,
                    class_names=CLASS_NAMES, progress: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated experiments. Every valid CAV is evaluated for all 10 target classes.

    Returns (cav_df: one row per CAV run, res_df: one row per valid CAV run x class).
    Invalid concept CAVs are recorded in ``cav_df`` but excluded from ``res_df``.
    """
    jobs = build_jobs(cfg)
    cav_rows, res_rows = [], []
    pbar = tqdm(total=sum(j[4] for j in jobs), desc="CAV runs", disable=not progress)
    for layer, concept, ci, kind, n_runs in jobs:
        for run in range(n_runs):
            cav = fit_cav_run(layer, concept, ci, run, kind, concept_feats, ref_feats, cfg)
            cav_rows.append({"layer": layer, "concept": concept, "kind": kind, "run": run,
                             "train_acc": cav["train_acc"], "val_acc": cav["val_acc"], "w_norm": cav["w_norm"],
                             "finite": cav["finite"], "orient_ok": cav["orient_ok"], "valid": cav["valid"]})
            if cav["valid"]:
                for k in range(NUM_CLASSES):
                    r = tcav_for_cav(grads, correct_mask, layer, cav["v"], k, cfg.tcav_only_correct)
                    res_rows.append({"layer": layer, "concept": concept, "kind": kind, "run": run,
                                     "class_idx": k, "class_name": class_names[k], **r})
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(cav_rows), pd.DataFrame(res_rows)
