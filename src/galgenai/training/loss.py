"""Reconstruction / flow-matching loss functions.

These operate in raw (de-normalized) flux units when a ``denorm_fn``
is provided, so the loss can be computed in physically meaningful
units and combined with inverse-variance weights.
``masked_weighted_mse`` builds on ``mse`` so both functions treat
de-normalization identically.
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F


def mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    denorm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Mean squared error, optionally computed in de-normalized units.

    Args:
        pred: Predicted tensor.
        target: Target tensor.
        denorm_fn: Optional denormalization function applied to both
            ``pred`` and ``target`` before computing the error (e.g. to
            convert normalized images back to flux units).
        reduction: Reduction passed to ``F.mse_loss`` ('mean', 'sum',
            or 'none').

    Returns:
        The (optionally reduced) squared error.
    """
    if denorm_fn is not None:
        pred = denorm_fn(pred)
        target = denorm_fn(target)
    return F.mse_loss(pred, target, reduction=reduction)


def masked_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    ivar: torch.Tensor,
    mask: torch.Tensor,
    denorm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    scale_by_total_pixels: bool = False,
) -> torch.Tensor:
    """
    Inverse-variance weighted MSE over valid (masked) pixels.

    Delegates the (optionally de-normalized) squared-error computation
    to :func:`mse` and then applies inverse-variance weighting and
    masking, averaging over the number of valid pixels.

    Args:
        pred: Predicted tensor.
        target: Target tensor.
        ivar: Inverse-variance weights, broadcastable to ``pred``.
        mask: Boolean/float mask indicating valid pixels.
        denorm_fn: Optional denormalization function applied to both
            ``pred`` and ``target`` before computing the error.
        scale_by_total_pixels: If True, multiply the per-valid-pixel
            mean by the total number of elements in ``pred``. This
            recovers a sum-like magnitude comparable to
            ``mse(reduction='sum')``.

    Returns:
        Scalar weighted MSE.
    """
    if ivar is None or mask is None:
        raise ValueError(
            "masked_weighted_mse requires both ivar and mask arguments"
        )

    # mse applies denorm consistently; "none" yields elementwise SE.
    squared_error = mse(pred, target, denorm_fn=denorm_fn, reduction="none")

    mask_float = mask.float()
    weighted_error = squared_error * ivar * mask_float
    num_valid = mask_float.sum().clamp(min=1.0)
    loss = weighted_error.sum() / num_valid

    if scale_by_total_pixels:
        # denorm_fn preserves shape, so pred.numel() is the count.
        loss = loss * pred.numel()

    return loss
