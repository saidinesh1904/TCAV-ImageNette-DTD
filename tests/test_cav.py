import numpy as np
import torch

from tcav_imagenette.cav import (check_orientation, fit_cav_run, make_cav_dataset, stratified_split,
                                 unit_normalize)
from tcav_imagenette.config import RANDOM_REF_INDEX, RANDOM_REF_NAME, Config
from tcav_imagenette.controls import build_jobs, run_experiments

D, N = 32, 40
LAYER = "layer3"


def _synthetic(shift=4.0, seed=0):
    """Concept features shifted by `shift` along a known direction u; reference features centred at 0."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(D, generator=g)
    u = u / u.norm()
    ref = torch.randn(200, D, generator=g)
    pos = torch.randn(N, D, generator=g) + shift * u
    return u, {LAYER: {"c": pos}}, {LAYER: ref}


def _cfg(**kw):
    base = dict(concepts=["c"], concept_class_hypotheses={}, tcav_layers=[LAYER], primary_layer=LAYER,
                n_concept_imgs=N, ref_pool_size=200, n_cav_runs=3, n_random_runs=3)
    base.update(kw)
    return Config(**base)


def test_dataset_is_balanced_with_concept_as_label_one():
    _, cf, rf = _synthetic()
    for kind in ("concept", "shuffled_labels", "random_reference"):
        X, y = make_cav_dataset(LAYER, "c", kind, np.random.default_rng(0), cf, rf, N)
        assert X.shape == (2 * N, D) and y.shape == (2 * N,) and int(y.sum()) == N
    X, y = make_cav_dataset(LAYER, "c", "concept", np.random.default_rng(0), cf, rf, N)
    assert torch.equal(X[:N], cf[LAYER]["c"]) and bool((y[:N] == 1).all()) and bool((y[N:] == 0).all())
    X, y = make_cav_dataset(LAYER, "c", "shuffled_labels", np.random.default_rng(0), cf, rf, N)
    assert not bool((y[:N] == 1).all())                       # labels were permuted


def test_stratified_split_is_disjoint_and_class_balanced():
    y = np.array([0] * 40 + [1] * 40)
    tr, va = stratified_split(y, 0.2, np.random.default_rng(0))
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == 80
    assert (y[va] == 1).sum() == (y[va] == 0).sum() == 8


def test_unit_normalize_and_orientation():
    v, norm, finite = unit_normalize(torch.tensor([3.0, 4.0]))
    assert finite and abs(norm - 5.0) < 1e-6 and abs(v.norm().item() - 1) < 1e-6
    _, _, finite0 = unit_normalize(torch.zeros(3))
    _, _, finite_nan = unit_normalize(torch.tensor([float("nan"), 1.0]))
    assert not finite0 and not finite_nan
    X = torch.tensor([[2.0, 0.0], [3.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]])
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert check_orientation(torch.tensor([1.0, 0.0]), X, y)
    assert not check_orientation(torch.tensor([-1.0, 0.0]), X, y)
    assert not check_orientation(torch.tensor([1.0, 0.0]), X, y, finite=False)


def test_true_concept_cav_is_unit_oriented_and_recovers_direction():
    u, cf, rf = _synthetic()
    r = fit_cav_run(LAYER, "c", 0, 0, "concept", cf, rf, _cfg())
    assert abs(r["v"].norm().item() - 1.0) < 1e-5
    assert r["finite"] and r["orient_ok"] and r["valid"] and r["val_acc"] >= 0.9
    assert float(r["v"] @ u) > 0.5                            # points toward increasing concept presence


def test_fit_cav_run_is_deterministic_and_layer_independent_seeding():
    _, cf, rf = _synthetic()
    a = fit_cav_run(LAYER, "c", 0, 1, "concept", cf, rf, _cfg())
    b = fit_cav_run(LAYER, "c", 0, 1, "concept", cf, rf, _cfg())
    assert torch.equal(a["v"], b["v"]) and a["val_acc"] == b["val_acc"]
    c = fit_cav_run(LAYER, "c", 0, 2, "concept", cf, rf, _cfg())
    assert not torch.equal(a["v"], c["v"])


def test_shuffled_label_control_is_near_chance_and_not_filtered_by_accuracy():
    _, cf, rf = _synthetic()
    r = fit_cav_run(LAYER, "c", 0, 0, "shuffled_labels", cf, rf, _cfg())
    assert r["val_acc"] < 0.85 and r["valid"] == r["finite"]


def test_low_accuracy_concept_cav_is_marked_invalid():
    _, cf, rf = _synthetic(shift=0.0)                         # concept == reference distribution
    r = fit_cav_run(LAYER, "c", 0, 0, "concept", cf, rf, _cfg(cav_min_acc=0.99))
    assert r["finite"] and not r["valid"]


def test_build_jobs_plan():
    cfg = Config(concepts=["a", "b"], concept_class_hypotheses={}, tcav_layers=["layer4", "layer3"],
                 primary_layer="layer3", n_cav_runs=3, n_random_runs=4)
    jobs = build_jobs(cfg)
    assert len(jobs) == 2 * (2 * 2 + 1)
    assert sum(j[4] for j in jobs if j[3] == "concept") == 2 * 2 * 3
    ref = [j for j in jobs if j[3] == "random_reference"]
    assert len(ref) == 2 and all(j[1] == RANDOM_REF_NAME and j[2] == RANDOM_REF_INDEX for j in ref)


def test_run_experiments_end_to_end_on_synthetic_features():
    _, cf, rf = _synthetic()
    cfg = _cfg()
    g = torch.Generator().manual_seed(3)
    grads = {LAYER: {k: torch.randn(7, D, generator=g) for k in range(10)}}
    mask = {k: torch.ones(7, dtype=torch.bool) for k in range(10)}
    cav_df, res_df = run_experiments(cfg, cf, rf, grads, mask, progress=False)
    assert len(cav_df) == 3 + 3 + 3                           # concept + shuffled_labels + random_reference
    assert set(cav_df.kind) == {"concept", "shuffled_labels", "random_reference"}
    assert set(res_df.kind) == {"concept", "shuffled_labels", "random_reference"}
    assert (res_df.groupby(["kind", "run"]).class_idx.nunique() == 10).all()   # every CAV scored on all classes
    assert res_df.tcav.between(0, 1).all()
    assert (res_df.n_images == 7).all()
