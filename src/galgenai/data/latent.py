"""Latent representation dataset for CNF training."""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from galgenai.models import VAEEncoder


class LatentDataset(Dataset):
    """Dataset for latent representations.

    Samples from VAE posterior. Loads mu and logvar from .pt cache file
    into memory and samples from N(mu, sigma^2) when accessed.

    Preserves stochasticity during CNF training.

    Parameters:
    -----------
    cache_path: Path to .pt cache with mu, logvar,
        and conditions tensors.

    """

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        # Load cached latents into memory
        data = torch.load(
            self.cache_path, map_location="cpu", weights_only=True
        )
        self.mu = data["mu"]
        self.logvar = data["logvar"]
        self.conditions = data["conditions"]

        self.latent_dim = self.mu.shape[1]
        self.condition_dim = self.conditions.shape[1]

    def __len__(self):
        return len(self.mu)

    def __getitem__(self, idx):
        """Sample from the latent posterior N(mu, sigma^2)."""
        mu = self.mu[idx]
        logvar = self.logvar[idx]
        cond = self.conditions[idx]

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, cond


@torch.no_grad()
def precompute_latents(
    encoder: VAEEncoder,
    loader: DataLoader,
    device: torch.device,
    cache_path: str | Path,
    return_dataloader: bool = False,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> LatentDataset | DataLoader:
    """Encode all images and cache to .pt file.

    LatentDataset samples from N(mu, sigma^2), preserving stochasticity.

    Always computes latents and writes to cache, overwriting any
    existing file.

    Parameters:
    -----------
    encoder: Frozen VAE encoder in eval mode.
    loader: DataLoader returning batches. Supports: (flux, condition) or
        (flux, ivar, mask, condition) tuples
    device: Device to run encoder on.
    cache_path: Path to .pt cache file. Overwritten if exists.
    return_dataloader: If True, return DataLoader instead of
        LatentDataset.
    batch_size: Batch size for returned DataLoader. If None, uses input
        loader.
    shuffle: Shuffle the DataLoader (default True).

    Returns:
    --------
    If return_dataloader=False: LatentDataset
    If return_dataloader=True: DataLoader
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print("Computing latents from encoder...")
    encoder.eval()

    all_mu: list[torch.Tensor] = []
    all_logvar: list[torch.Tensor] = []
    all_cond: list[torch.Tensor] = []

    for batch in loader:
        # Handle both 2-tuple and 4-tuple formats
        # flux is always first, condition is always last
        images = batch[0].to(device)
        cond = batch[-1]

        mu, logvar = encoder(images)
        all_mu.append(mu.cpu())
        all_logvar.append(logvar.cpu())
        all_cond.append(cond.cpu())

    # Concatenate all batches
    mu = torch.cat(all_mu, dim=0)
    logvar = torch.cat(all_logvar, dim=0)
    conditions = torch.cat(all_cond, dim=0)

    print(f"Computed {len(mu)} latents")

    # Save to cache
    print(f"Saving to cache: {cache_path}")
    torch.save(
        {
            "mu": mu,
            "logvar": logvar,
            "conditions": conditions,
        },
        cache_path,
    )
    print("Cache saved successfully")

    # Create dataset from cache
    dataset = LatentDataset(cache_path=cache_path)

    if return_dataloader:
        if batch_size is None:
            batch_size = loader.batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )
    else:
        return dataset
