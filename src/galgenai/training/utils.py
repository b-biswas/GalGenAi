"""Shared training utilities."""

from typing import Callable, Optional, Tuple

import torch

from .loss import masked_weighted_mse, mse


def extract_batch_data(
    batch, device: torch.device, extract_noiseless: bool = False
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Extract data from batch.

    HSCDataset always returns 5-tuple:
    (flux, ivar, mask, noiseless_flux, condition)
    where non-requested values are None.

    Args:
        batch: 5-tuple from HSCDataset.
        device: Device to move tensors to.
        extract_noiseless: If True, extract noiseless flux.

    Returns:
        Tuple of (data, ivar, mask, noiseless_flux). ivar, mask, and
        noiseless_flux may be None.
    """
    # HSCDataset always returns (flux, ivar, mask, noiseless, cond)
    flux, ivar, mask, noiseless_flux, _ = batch

    data = flux.to(device)
    ivar = ivar.to(device) if ivar is not None else None
    mask = mask.to(device) if mask is not None else None

    if extract_noiseless and noiseless_flux is not None:
        noiseless_flux = noiseless_flux.to(device)
    else:
        noiseless_flux = None

    return data, ivar, mask, noiseless_flux


def vae_loss(
    reconstruction: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reconstruction_loss_fn: str = "mse",
    beta: float = 1.0,
    ivar: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    denorm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate VAE loss = reconstruction_loss + beta * KL_divergence.

    Args:
        reconstruction: Reconstructed images.
        x: Original images.
        mu: Mean of latent distribution.
        logvar: Log variance of latent distribution.
        reconstruction_loss_fn: Type of reconstruction loss
            ('mse' or 'masked_weighted_mse').
        beta: Weight for KL divergence term (beta-VAE).
        ivar: Inverse variance weights for each pixel (optional).
        mask: Boolean mask indicating valid pixels (optional).
        denorm_fn: Optional denormalization function to convert
            normalized images back to flux units before computing loss.
            Only used with 'masked_weighted_mse'.

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence).
    """
    if reconstruction_loss_fn == "mse":
        recon_loss = mse(reconstruction, x, reduction="sum")
    elif reconstruction_loss_fn == "masked_weighted_mse":
        recon_loss = masked_weighted_mse(
            reconstruction,
            x,
            ivar,
            mask,
            denorm_fn=denorm_fn,
            scale_by_total_pixels=True,
        )
    else:
        raise ValueError(
            f"Unknown reconstruction loss: {reconstruction_loss_fn}"
        )

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss, kl_div
