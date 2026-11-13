import numpy as np
import torch

from tcav_imagenette.tcav import (directional_derivative_autograd, directional_derivative_fd,
                                  forward_from_pooled, pooled_gradient, summarize_derivatives,
                                  tcav_for_cav, verify_directional_derivative)


def _unit(d, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g, dtype=torch.float64)
    return v / v.norm()


def test_intervention_is_zero_in_value(tiny_model, tiny_split, images):
    with torch.no_grad():
        for layer in ("layer3", "layer4"):
            A = tiny_split.trunk(images, layer)
            logits = forward_from_pooled(tiny_split, A, A.mean(dim=(2, 3)), layer)
            assert torch.allclose(logits, tiny_model(images), atol=1e-9, rtol=1e-9)


def test_layer4_gradient_is_the_linear_head_row(tiny_model, tiny_split, images):
    """Pooled layer4 -> linear head: grad of logit k is w_k for every image."""
    A = tiny_split.trunk(images, "layer4").detach()
    for k in (0, 3, 9):
        g, _ = pooled_gradient(tiny_split, A, k, "layer4")
        w_k = tiny_model.fc.weight[k].detach()
        assert torch.allclose(g, w_k.expand_as(g), atol=1e-9)


def test_layer3_gradient_varies_across_images_layer4_does_not(tiny_split, images):
    rel = {}
    for layer in ("layer3", "layer4"):
        A = tiny_split.trunk(images, layer).detach()
        g, _ = pooled_gradient(tiny_split, A, 2, layer)
        rel[layer] = (g.std(dim=0).mean() / (g.abs().mean() + 1e-12)).item()
    assert rel["layer4"] < 1e-9
    assert rel["layer3"] > 1e-3


def test_autograd_scalar_eps_and_finite_difference_agree(tiny_split, images):
    for layer in ("layer3", "layer4"):
        A = tiny_split.trunk(images, layer).detach()
        v = _unit(A.shape[1], seed=5)
        g, _ = pooled_gradient(tiny_split, A, 4, layer)
        s_a = (g * v.view(1, -1)).sum(1)
        s_b = directional_derivative_autograd(tiny_split, A, 4, v, layer)
        s_c = directional_derivative_fd(tiny_split, A, 4, v, layer, eps=1e-6)
        scale = s_a.abs().max().item() + 1e-12
        assert (s_a - s_b).abs().max().item() < 1e-9 * max(1.0, scale)
        assert (s_a - s_c).abs().max().item() < 1e-5 * max(1.0, scale)


def test_verify_directional_derivative_reports_small_differences(tiny_split, images):
    layer, k = "layer3", 1
    A = tiny_split.trunk(images, layer).detach()
    g, _ = pooled_gradient(tiny_split, A, k, layer)
    v = _unit(A.shape[1], seed=6)
    res = verify_directional_derivative(tiny_split, images, layer, k, v, stored_grads=g, eps=1e-6)
    assert set(res) == {"max_abs_a_b", "max_abs_a_c", "scale_S", "max_abs_a_d"}
    assert res["max_abs_a_b"] < 1e-8 and res["max_abs_a_d"] < 1e-8
    assert res["max_abs_a_c"] < 1e-5 * max(1.0, res["scale_S"])


def test_summarize_derivatives_counts_strictly_positive_only():
    r = summarize_derivatives(torch.tensor([1.0, -1.0, 0.0, 2.0]))
    assert r["n_images"] == 4 and r["n_pos"] == 2 and r["n_neg"] == 1
    assert r["tcav"] == 0.5 and r["frac_neg"] == 0.25          # exact zero is non-positive
    empty = summarize_derivatives(torch.tensor([]))
    assert empty["n_images"] == 0 and np.isnan(empty["tcav"])


def test_tcav_for_cav_fraction_of_positive_directional_derivatives():
    # gradients (4 images, 2 dims); v = e_1 -> S = first column
    G = torch.tensor([[1.0, 5.0], [-2.0, 5.0], [3.0, -5.0], [0.5, 0.0]])
    grads = {"layer3": {7: G}}
    mask = {7: torch.tensor([True, False, True, False])}
    v = torch.tensor([1.0, 0.0])
    r = tcav_for_cav(grads, mask, "layer3", v, 7)
    assert r["tcav"] == 0.75 and r["n_images"] == 4
    r_ok = tcav_for_cav(grads, mask, "layer3", v, 7, only_correct=True)
    assert r_ok["n_images"] == 2 and r_ok["tcav"] == 1.0
    r_neg = tcav_for_cav(grads, mask, "layer3", -v, 7)
    assert abs(r_neg["tcav"] - 0.25) < 1e-12


def test_layer4_tcav_is_zero_or_one_for_a_single_cav(tiny_model, tiny_split, images):
    """Image-independent layer4 gradient => S has one sign for all images => TCAV in {0, 1}."""
    A = tiny_split.trunk(images, "layer4").detach()
    g, _ = pooled_gradient(tiny_split, A, 0, "layer4")
    for seed in range(4):
        r = tcav_for_cav({"layer4": {0: g.float()}}, {0: torch.ones(len(g), dtype=torch.bool)},
                         "layer4", _unit(A.shape[1], seed), 0)
        assert r["tcav"] in (0.0, 1.0)
