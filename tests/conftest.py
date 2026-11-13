"""Shared synthetic fixtures (no datasets, no pretrained weights, no checkpoints needed)."""
import pytest
import torch
import torch.nn as nn

from tcav_imagenette.features import SplitResNet
from tcav_imagenette.model import build_model, freeze_model


@pytest.fixture(scope="module")
def tiny_model():
    """Randomly initialised 10-class ResNet50 in float64 with calibrated BatchNorm statistics."""
    torch.manual_seed(0)
    model = build_model(pretrained=False).double()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = None                       # cumulative average -> running stats = batch stats
    model.train()
    with torch.no_grad():
        model(torch.randn(8, 3, 64, 64, dtype=torch.float64))
    return freeze_model(model)


@pytest.fixture(scope="module")
def tiny_split(tiny_model):
    return SplitResNet(tiny_model)


@pytest.fixture(scope="module")
def images():
    g = torch.Generator().manual_seed(1)
    return torch.randn(4, 3, 64, 64, generator=g, dtype=torch.float64)
