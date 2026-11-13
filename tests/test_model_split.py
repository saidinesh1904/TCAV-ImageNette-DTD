import torch

from tcav_imagenette.config import CLASS_NAMES, LAYER_DIMS, NUM_CLASSES, Config
from tcav_imagenette.features import check_split
from tcav_imagenette.model import build_model, freeze_model, train_fingerprint


def test_build_model_is_ten_class():
    m = build_model(pretrained=False)
    assert m.fc.in_features == 2048 and m.fc.out_features == NUM_CLASSES == len(CLASS_NAMES)


def test_freeze_model():
    m = freeze_model(build_model(pretrained=False))
    assert not m.training and all(not p.requires_grad for p in m.parameters())


def test_split_shapes_and_equivalence(tiny_model, tiny_split, images):
    res = check_split(tiny_model, tiny_split, images, ["layer3", "layer4"], torch.device("cpu"),
                      atol_act=1e-9, atol_logits=1e-9, rtol_logits=1e-9)
    assert res["layer3"]["activation_shape"] == (4, LAYER_DIMS["layer3"], 4, 4)
    assert res["layer4"]["activation_shape"] == (4, LAYER_DIMS["layer4"], 2, 2)
    for layer in ("layer3", "layer4"):
        assert res[layer]["pooled_shape"] == (4, LAYER_DIMS[layer])
        assert res[layer]["matches_hook"]
        assert res[layer]["downstream_matches_model"]


def test_trunk_multi_matches_single_trunk(tiny_split, images):
    with torch.no_grad():
        both = tiny_split.trunk_multi(images, ["layer4", "layer3"])
        assert set(both) == {"layer3", "layer4"}
        assert torch.equal(both["layer3"], tiny_split.trunk(images, "layer3"))
        assert torch.equal(both["layer4"], tiny_split.trunk(images, "layer4"))


def test_downstream_from_every_layer_reproduces_logits(tiny_model, tiny_split, images):
    with torch.no_grad():
        ref = tiny_model(images)
        for layer in ("layer1", "layer2", "layer3", "layer4"):
            out = tiny_split.downstream(tiny_split.trunk(images, layer), layer)
            assert torch.allclose(out, ref, atol=1e-9, rtol=1e-9), layer


def test_train_fingerprint_depends_on_training_settings():
    a, b = Config(), Config(epochs=2)
    assert train_fingerprint(a, 100) != train_fingerprint(b, 100)
    assert train_fingerprint(a, 100) != train_fingerprint(a, 101)
    assert train_fingerprint(a, 100) == train_fingerprint(Config(), 100)
