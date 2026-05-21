from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from datasets import Dataset
from torch.utils.data import DataLoader, random_split

from galgenai.data.hsc import HSCDataset


# TODO: CHECK catalog what the sentinel values are
def load_fits_dataset(
    data_dir,
    metadata_file="metadata.csv",
    format="torch",
    filter_invalid_mags=True,
    mag_sentinel=999.0,
    mag_cols=None,
    filter_invalid_redshift=True,
    redshift_sentinel=-99.0,
    redshift_col=None,
    nx=None,
):
    """
    Load a FITS galaxy dataset from generate_fits_dataset.

    All galaxies stored under ``data_dir/images/``
    with metadata CSV. Returned dataset is split in
    make_loaders().

    Dataset has one column image (dict with keys:
    flux, ivar, mask, band) plus metadata from
    CSV. Matches HSC dataset format for use with
    hsc.HSCDataset.

    If IVAR absent, ones array (uniform weight).
    If MASK absent, zeros array (no masking).

    Parameters:
    -----------
    data_dir : str or Path
        Root directory (contains ``images/`` and
        metadata CSV).
    metadata_file : str or dict
        If str: Metadata CSV filename ("metadata.csv").
        If dict: Maps split names to filenames.
                 Returns dict of datasets.
    format : str
        Output format: "torch" (default), "numpy",
        "tensorflow", or None. Default "torch".
    filter_invalid_mags : bool
        If True, filter galaxies with sentinel
        mag values. Default True.
    mag_sentinel : float
        Sentinel value for missing magnitude
        (default: 999.0).
    mag_cols : list of str or None
        Magnitude column names to check.
        If None, auto-detects. Default None.
    filter_invalid_redshift : bool
        If True, filter sentinel redshift.
        Default True.
    redshift_sentinel : float
        Sentinel value for missing redshift
        (default: -99.0).
    redshift_col : str
        Redshift column name in metadata.
    nx : int or None
        Crop size. If None, original size.
        Default None.

    Returns:
    --------
    datasets.Dataset or dict
        HuggingFace Dataset with PyTorch tensors
        (default format="torch").
        If metadata_file is str: Single dataset.
        If metadata_file is dict: Dict of datasets.
    """
    data_dir = Path(data_dir)

    # Handle dict of metadata files (train/test/val splits)
    if isinstance(metadata_file, dict):
        result = {}
        for split_name, meta_file in metadata_file.items():
            result[split_name] = load_fits_dataset(
                data_dir,
                metadata_file=meta_file,
                format=format,
                filter_invalid_mags=filter_invalid_mags,
                mag_sentinel=mag_sentinel,
                mag_cols=mag_cols,
                filter_invalid_redshift=filter_invalid_redshift,
                redshift_sentinel=redshift_sentinel,
                redshift_col=redshift_col,
                nx=nx,
            )
        return result

    # Single metadata file
    images_path = data_dir / "images"
    metadata = pd.read_csv(data_dir / metadata_file)

    # Filter out galaxies with invalid magnitudes
    if filter_invalid_mags:
        initial_count = len(metadata)

        # Determine which columns to check
        if mag_cols is None:
            # Auto-detect magnitude columns
            raise ValueError("Mag col names should be provided to apply cuts")

        if mag_cols:
            # Create mask for valid magnitudes
            mask = np.ones(len(metadata), dtype=bool)
            for mag_col in mag_cols:
                if mag_col in metadata.columns:
                    # Filter rows with sentinel magnitude
                    col_mask = metadata[mag_col] <= mag_sentinel
                    mask &= col_mask

            metadata = metadata[mask].reset_index(drop=True)
            n_removed = initial_count - len(metadata)
            if n_removed > 0:
                print(
                    f"Filtered {n_removed} galaxies with "
                    f"invalid magnitudes "
                    f"(sentinel={mag_sentinel})"
                )
                print(f"Remaining: {len(metadata)} galaxies")

    if filter_invalid_redshift:
        if redshift_col is None:
            raise ValueError(
                "Redshift col name should be provided to apply cuts"
            )

        if redshift_col in metadata.columns:
            col_mask = metadata[redshift_col] != redshift_sentinel
            metadata = metadata[col_mask].reset_index(drop=True)
            n_removed = len(col_mask) - len(metadata)
            if n_removed > 0:
                print(
                    f"Filtered {n_removed} galaxies with "
                    f"invalid redshift "
                    f"(sentinel={redshift_sentinel})"
                )
                print(f"Remaining: {len(metadata)} galaxies")
        else:
            raise ValueError(
                f"Redshift column '{redshift_col}' not found in metadata."
            )

    # Check for Arrow cache (memory-mappable)
    from datasets import load_from_disk

    # Cache path includes crop size
    cache_suffix = ""
    if nx is not None:
        cache_suffix += f"_nx{nx}"

    cache_name = (
        f"arrow_cache{cache_suffix}" if cache_suffix else "arrow_cache_raw"
    )
    cache_path = data_dir / cache_name

    if cache_path.exists():
        print(f"Loading from Arrow cache: {cache_path}")
        dataset = load_from_disk(str(cache_path))
    else:
        print(
            f"Arrow cache not found. Loading {len(metadata):,} FITS files..."
        )
        print("This ONE-TIME operation will take a few minutes.")

        if nx is not None:
            print(
                f"Images will be center-cropped to {nx}x{nx} during caching."
            )
        else:
            print("Images will be cached at their original size.")

        from tqdm import tqdm

        n_total = len(metadata)

        # Get original image size and band names from first FITS file
        first_row = metadata.iloc[0]
        with fits.open(images_path / first_row["filename"]) as hdul:
            orig_shape = hdul["IMAGE"].data.shape
            og_h, og_w = orig_shape[1], orig_shape[2]

            # Extract band names from FITS header (same for all images)
            n_bands = orig_shape[0]
            bands = [
                hdul["IMAGE"].header.get(f"BAND{i}", f"band{i}")
                for i in range(n_bands)
            ]

        if nx is not None:
            # Calculate crop indices
            og_nx2, og_ny2 = og_h // 2, og_w // 2
            nx2 = nx // 2
            print(f"  - Original size: {og_h}x{og_w}, cropped to: {nx}x{nx}")

        # Process all samples in ONE pass
        print(f"Processing {n_total:,} samples...")
        all_samples = []

        for i in tqdm(range(n_total), desc="Loading FITS files"):
            row = metadata.iloc[i]
            with fits.open(images_path / row["filename"]) as hdul:
                # Load raw data
                flux = hdul["IMAGE"].data.astype("float32")

                if "IVAR" in hdul:
                    ivar = hdul["IVAR"].data.astype("float32")
                else:
                    ivar = np.ones_like(flux)

                if "MASK" in hdul:
                    mask = hdul["MASK"].data.astype(np.int32)
                else:
                    mask = np.zeros(flux.shape, dtype=np.int32)

            # Crop to target size if nx is provided
            if nx is not None:
                flux = flux[
                    :, og_nx2 - nx2 : og_nx2 + nx2, og_ny2 - nx2 : og_ny2 + nx2
                ]
                ivar = ivar[
                    :, og_nx2 - nx2 : og_nx2 + nx2, og_ny2 - nx2 : og_ny2 + nx2
                ]
                mask = mask[
                    :, og_nx2 - nx2 : og_nx2 + nx2, og_ny2 - nx2 : og_ny2 + nx2
                ]

            sample = {
                "image": {
                    "flux": flux,
                    "ivar": ivar,
                    "mask": mask,
                    "band": bands,
                }
            }

            sample.update(row.to_dict())

            all_samples.append(sample)

        print("Creating Arrow cache...")
        dataset = Dataset.from_list(all_samples)
        del all_samples

        # Save to disk with sharding for optimal memory mapping
        print(f"Saving to: {cache_path}")
        dataset.save_to_disk(str(cache_path))
        cache_size_mb = (
            sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            / 1e6
        )
        print(
            f"Cached! ({cache_size_mb:.1f} MB) Future loads will be instant."
        )

    # Apply format if specified
    if format is not None:
        dataset = dataset.with_format(format)

    return dataset


def make_loaders(
    dataset_raw: Dataset,
    nx: int,
    batch_size: int,
    num_workers: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    image_norm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    return_aux_data: bool = True,
    condition_cols: Optional[list] = None,
    conditional_norm_fn: Optional[
        Callable[[torch.Tensor], torch.Tensor]
    ] = None,
    shuffle: bool = True,
    split_datasets: Optional[tuple] = None,
    return_splits: bool = False,
    invert_mask: bool = False,
):
    """Build DataLoaders from dataset.

    Takes a raw dataset and splits it into
    train/val/test, wraps each in HSCDataset,
    and creates dataloaders.

    Supports multiple data modes:
    1. VAE training: return_aux_data=True
       Returns (flux, ivar, mask) tuples.
    2. Latent precomputation: with conditioning
       Returns (flux, condition) tuples.
    3. CNF training: return_aux_data=False
       Returns (flux, condition) tuples.

    Parameters:
    -----------
    dataset_raw: Raw HuggingFace Dataset
    nx: Side length of center-cropped patch
    batch_size: Batch size for DataLoaders
    num_workers: Number of worker processes
    train_ratio: Fraction for training (default 0.8)
    val_ratio: Fraction for validation (default 0.1)
    random_seed: Random seed for splits (default 42)
    image_norm_fn: Optional normalization function
        from get_image_norm_fn()
    return_aux_data: If True, return
        (flux, ivar, mask) else (flux, condition)
    condition_cols: Optional conditioning columns
    conditional_norm_fn: Optional normalization
        from get_conditional_norm_fn()
    shuffle: Shuffle training data (default True)
    split_datasets: Tuple of (train, val, test)
        from previous call
    return_splits: If True, return split datasets
        (default False)
    invert_mask: If True, flip per-pixel mask
        emitting it. Set this when the source survey writes
        ``1 = bad pixel`` rather than the ``1 = valid pixel`` convention
        the trainers assume. Default False.

    Returns:
    --------
    If return_splits=False: (train_loader, val_loader, test_loader)
        where test_loader is None if train_ratio + val_ratio == 1.0
    If return_splits=True:
        (train_loader, val_loader, test_loader, split_datasets)
        where split_datasets is a tuple (train_ds, val_ds, test_ds)
        that can be passed to subsequent calls to reuse the same split
        with different normalization.
    """
    # Validate conditioning parameters
    if condition_cols is not None and conditional_norm_fn is None:
        raise ValueError(
            "conditional_norm_fn must be provided when using condition_cols. "
            "Use get_conditional_norm_fn() to create it."
        )

    # Determine if pin_memory should be used (only supported on CUDA)
    use_pin_memory = torch.cuda.is_available()

    # Create HSCDataset wrapper for the full dataset
    full_dataset = HSCDataset(
        dataset_raw,
        nx=nx,
        image_norm_fn=image_norm_fn,
        return_aux_data=return_aux_data,
        condition_cols=condition_cols or [],
        conditional_norm_fn=conditional_norm_fn,
        invert_mask=invert_mask,
    )

    # Split dataset using random_split or reuse existing split
    if split_datasets is not None:
        # Reuse existing split by extracting indices and new Subsets
        old_train, old_val, old_test = split_datasets
        train_ds = torch.utils.data.Subset(full_dataset, old_train.indices)
        val_ds = torch.utils.data.Subset(full_dataset, old_val.indices)
        test_ds = torch.utils.data.Subset(full_dataset, old_test.indices)
    else:
        # Create new split
        test_ratio = 1.0 - train_ratio - val_ratio
        train_ds, val_ds, test_ds = random_split(
            full_dataset,
            [train_ratio, val_ratio, test_ratio],
            generator=torch.Generator().manual_seed(random_seed),
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True
        if num_workers > 0
        else False,  # Reuse worker processes across epochs
        prefetch_factor=num_workers * 4
        if num_workers > 0
        else None,  # Prefetch batches
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,  # Validation is never shuffled
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=num_workers * 4 if num_workers > 0 else None,
    )

    # Create test loader if test set exists
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,  # Test is never shuffled
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=num_workers * 4 if num_workers > 0 else None,
        )
    else:
        test_loader = None

    if return_splits:
        return (
            train_loader,
            val_loader,
            test_loader,
            (train_ds, val_ds, test_ds),
        )
    else:
        return train_loader, val_loader, test_loader
