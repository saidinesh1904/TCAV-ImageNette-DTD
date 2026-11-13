import json
from pathlib import Path

import pytest

from tcav_imagenette.config import CLASS_NAMES, Config, load_config, validate_config

DEFAULT_JSON = Path(__file__).resolve().parents[1] / "configs" / "default.json"


def test_default_json_matches_builtin_defaults():
    assert load_config(DEFAULT_JSON).to_dict() == validate_config(Config()).to_dict()
    assert set(json.load(open(DEFAULT_JSON))) == set(Config().to_dict())


def test_experiment_design_is_unchanged():
    cfg = load_config(DEFAULT_JSON)
    assert cfg.primary_layer == "layer3" and cfg.tcav_layers == ["layer4", "layer3"]
    assert cfg.concept_class_hypotheses == {"dotted": "golf ball", "striped": "English springer",
                                            "banded": "parachute", "zigzagged": "chain saw",
                                            "woven": "cassette player"}
    assert cfg.concepts == list(cfg.concept_class_hypotheses)
    assert cfg.controls == ["shuffled_labels", "random_reference"]
    assert (cfg.n_cav_runs, cfg.n_random_runs, cfg.epochs, cfg.seed) == (30, 50, 6, 42)
    assert (cfg.cav_steps, cfg.cav_lr, cfg.cav_l2, cfg.cav_val_frac, cfg.cav_min_acc) == (200, 0.02, 1e-2, 0.2, 0.7)
    assert len(CLASS_NAMES) == 10


def test_pilot_overrides_only_epochs_and_run_counts():
    cfg = load_config(DEFAULT_JSON, {"pilot": True})
    assert (cfg.epochs, cfg.n_cav_runs, cfg.n_random_runs) == (2, 5, 10)
    assert cfg.lr == 1e-4 and cfg.primary_layer == "layer3"


def test_overrides_ignore_none_and_unknown_keys_fail():
    assert load_config(None, {"out_dir": None}).out_dir == "tcav_outputs"
    with pytest.raises(ValueError):
        load_config(None, {"not_a_setting": 1})


@pytest.mark.parametrize("bad", [
    {"primary_layer": "layer2"},
    {"tcav_layers": ["layer3", "layer9"]},
    {"concept_class_hypotheses": {"dotted": "unicorn"}},
    {"concept_class_hypotheses": {"plaid": "golf ball"}},
    {"controls": ["nonsense"]},
    {"reference_source": "elsewhere"},
    {"ref_pool_size": 100},
    {"alpha": 1.5},
])
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        load_config(None, bad)
