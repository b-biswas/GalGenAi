"""Tests for the shared loss functions in galgenai.training.loss."""

import pytest
import torch
import torch.nn.functional as F

from galgenai.training.loss import masked_weighted_mse, mse


def test_mse_matches_functional_mse():
    pred = torch.randn(4, 3, 8, 8)
    target = torch.randn(4, 3, 8, 8)

    for reduction in ("mean", "sum", "none"):
        expected = F.mse_loss(pred, target, reduction=reduction)
        assert torch.allclose(mse(pred, target, reduction=reduction), expected)


def test_mse_applies_denorm_to_both_args():
    pred = torch.randn(2, 1, 4, 4)
    target = torch.randn(2, 1, 4, 4)

    def denorm(x):
        return 2.0 * x + 1.0

    expected = F.mse_loss(denorm(pred), denorm(target))
    assert torch.allclose(mse(pred, target, denorm_fn=denorm), expected)


def test_masked_weighted_mse_matches_reference():
    pred = torch.randn(2, 1, 4, 4)
    target = torch.randn(2, 1, 4, 4)
    ivar = torch.rand(2, 1, 4, 4) + 0.1
    mask = (torch.rand(2, 1, 4, 4) > 0.3).float()

    mask_float = mask.float()
    weighted = (pred - target).pow(2) * ivar * mask_float
    expected = weighted.sum() / mask_float.sum().clamp(min=1.0)

    result = masked_weighted_mse(pred, target, ivar, mask)
    assert torch.allclose(result, expected)


def test_masked_weighted_mse_builds_on_mse():
    """The masked loss should equal mse(reduction='none') weighting."""
    pred = torch.randn(2, 1, 4, 4)
    target = torch.randn(2, 1, 4, 4)
    ivar = torch.rand(2, 1, 4, 4) + 0.1
    mask = (torch.rand(2, 1, 4, 4) > 0.3).float()

    se = mse(pred, target, reduction="none")
    weighted = se * ivar * mask.float()
    expected = weighted.sum() / mask.float().sum().clamp(min=1.0)

    assert torch.allclose(
        masked_weighted_mse(pred, target, ivar, mask), expected
    )


def test_masked_weighted_mse_applies_denorm():
    pred = torch.randn(2, 1, 4, 4)
    target = torch.randn(2, 1, 4, 4)
    ivar = torch.rand(2, 1, 4, 4) + 0.1
    mask = torch.ones(2, 1, 4, 4)

    def denorm(x):
        return 3.0 * x - 0.5

    weighted = (denorm(pred) - denorm(target)).pow(2) * ivar * mask
    expected = weighted.sum() / mask.sum().clamp(min=1.0)

    result = masked_weighted_mse(pred, target, ivar, mask, denorm_fn=denorm)
    assert torch.allclose(result, expected)


def test_masked_weighted_mse_scale_by_total_pixels():
    pred = torch.randn(2, 1, 4, 4)
    target = torch.randn(2, 1, 4, 4)
    ivar = torch.rand(2, 1, 4, 4) + 0.1
    mask = (torch.rand(2, 1, 4, 4) > 0.3).float()

    base = masked_weighted_mse(pred, target, ivar, mask)
    scaled = masked_weighted_mse(
        pred, target, ivar, mask, scale_by_total_pixels=True
    )
    assert torch.allclose(scaled, base * pred.numel())


def test_masked_weighted_mse_fully_masked_is_finite():
    pred = torch.randn(2, 1, 4, 4)
    target = torch.randn(2, 1, 4, 4)
    ivar = torch.rand(2, 1, 4, 4) + 0.1
    mask = torch.zeros(2, 1, 4, 4)

    result = masked_weighted_mse(pred, target, ivar, mask)
    assert torch.isfinite(result)
    assert result.item() == 0.0


def test_masked_weighted_mse_requires_ivar_and_mask():
    pred = torch.randn(2, 1, 4, 4)
    target = torch.randn(2, 1, 4, 4)
    mask = torch.ones(2, 1, 4, 4)

    with pytest.raises(ValueError):
        masked_weighted_mse(pred, target, None, mask)
    with pytest.raises(ValueError):
        masked_weighted_mse(pred, target, mask, None)
