import numpy as np
import pandas as pd

from tcav_imagenette.config import CLASS_NAMES, Config
from tcav_imagenette.statistics import (add_corrections, cliffs_delta, run_statistical_analysis, safe_mwu,
                                        safe_welch)


def test_cliffs_delta():
    assert cliffs_delta([2, 3, 4], [0, 1]) == 1.0
    assert cliffs_delta([0, 1], [2, 3, 4]) == -1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0


def test_safe_tests_handle_degenerate_inputs():
    assert np.isnan(safe_mwu([1.0], [0.0, 1.0])) and np.isnan(safe_welch([1.0], [0.0, 1.0]))
    assert safe_mwu([1.0] * 5, [1.0] * 5) == 1.0               # identical constant groups
    assert np.isnan(safe_welch([1.0] * 5, [1.0] * 5))          # zero variance in both groups
    assert safe_mwu(list(range(10, 20)), list(range(10))) < 0.01


def test_holm_and_bh_match_hand_computation():
    p = [0.01, 0.04, 0.03]
    df = pd.DataFrame({"layer": "layer3", "control": "shuffled_labels", "p_mwu_tcav": p,
                       "p_welch_tcav": p, "p_mwu_meanS": p, "diff_mean": [0.1, -0.1, 0.0]})
    out = add_corrections(df, alpha=0.05)
    assert np.allclose(out["p_mwu_tcav_holm"], [0.03, 0.06, 0.06])
    assert np.allclose(out["p_mwu_tcav_bh"], [0.03, 0.04, 0.04])
    assert list(out["supported_holm"]) == [True, False, False]
    assert list(out["direction"]) == ["above control", "below control", "equal"]


def test_families_are_corrected_separately():
    df = pd.DataFrame({"layer": ["layer3", "layer4"], "control": "shuffled_labels", "p_mwu_tcav": [0.02, 0.02],
                       "p_welch_tcav": [0.02, 0.02], "p_mwu_meanS": [0.02, 0.02], "diff_mean": [1.0, 1.0]})
    assert np.allclose(add_corrections(df, 0.05)["p_mwu_tcav_holm"], [0.02, 0.02])


def test_run_statistical_analysis_confirmatory_vs_exploratory():
    cfg = Config(concepts=["dotted"], concept_class_hypotheses={"dotted": "golf ball"},
                 tcav_layers=["layer3"], primary_layer="layer3", controls=["shuffled_labels"])
    rows = []
    for k in range(10):
        for run in range(10):
            rows.append(dict(layer="layer3", kind="concept", concept="dotted", run=run, class_idx=k,
                             class_name=CLASS_NAMES[k], tcav=0.8 + 0.01 * run, mean_S=1.0 + 0.1 * run))
            rows.append(dict(layer="layer3", kind="shuffled_labels", concept="dotted", run=run, class_idx=k,
                             class_name=CLASS_NAMES[k], tcav=0.4 + 0.01 * run, mean_S=0.1 * run))
    raw, expl, conf = run_statistical_analysis(pd.DataFrame(rows), cfg)
    assert len(raw) == len(expl) == 10 and len(conf) == 1
    assert conf.iloc[0].class_name == "golf ball" and bool(conf.iloc[0].is_hypothesis)
    assert bool(conf.iloc[0].supported_holm) and conf.iloc[0].direction == "above control"
    assert conf.iloc[0].cliffs_delta == 1.0
    # Holm over the 10 exploratory pairs multiplies p by 10 (all p are identical here)
    assert np.allclose(expl["p_mwu_tcav_holm"], np.minimum(1.0, expl["p_mwu_tcav"] * 10))
