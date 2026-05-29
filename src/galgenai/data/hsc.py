from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from datasets import Dataset

from .augmentation import random_rotation_and_flip


class HSCDataset(torch.utils.data.Dataset):
    """Unified dataset for galaxy images.

    For HSC/COSMOS with optional conditioning.

    Returns different data based on parameters:
    - return_aux_data=True: (flux, ivar, mask)
    - return_aux_data=False: flux
    - With conditioning: appends (condition)

    Mask convention: ``1 = valid, 0 = invalid``.

    Mask convention: downstream losses (VAE/LCFM/CFM weighted MSE) treat the
    emitted mask as ``1 = valid, 0 = invalid``. Set ``invert_mask=True`` when
    the source survey writes the opposite convention (``1 = bad pixel flag``).

    Args:
        hf_dataset: HuggingFace Dataset with 'image' column
        nx: Side length of center-cropped patch
        image_norm_fn: Optional normalization
        return_aux_data: Return auxiliary data
        condition_cols: Optional column names
        conditional_norm_fn: Optional function
        invert_mask: If True, flip mask
        augment: If True, apply random rotations and flips
    """

    def __init__(
        self,
        hf_dataset,
        nx: int,
        image_norm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_aux_data: bool = True,
        condition_cols: Optional[list] = None,
        conditional_norm_fn: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        invert_mask: bool = False,
        augment: bool = False,
    ):
        self.dataset = hf_dataset
        self.nx = nx
        self.image_norm_fn = image_norm_fn
        self.return_aux_data = return_aux_data
        self.condition_cols = condition_cols or []
        self.conditional_norm_fn = conditional_norm_fn
        self.invert_mask = invert_mask
        self.augment = augment

        # crop
        self.og_nx2 = self.dataset[0]["image"]["flux"].shape[1] // 2
        self.og_ny2 = self.dataset[0]["image"]["flux"].shape[2] // 2
        self.nx2 = nx // 2

        # bands
        self.bands = self.dataset[0]["image"]["band"]
        self.n_bands = self.dataset[0]["image"]["flux"].shape[0]

    def __len__(self):
        return len(self.dataset)

    def normalize(self, img):
        if self.image_norm_fn is not None:
            return self.image_norm_fn(img)
        return img

    def crop(self, img):
        return img[
            :,
            self.og_nx2 - self.nx2 : self.og_nx2 + self.nx2,
            self.og_ny2 - self.nx2 : self.og_ny2 + self.nx2,
        ]

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image_data = sample["image"]

        # Extract and crop flux
        flux = self.crop(image_data["flux"])

        # Process auxiliary data if requested
        if self.return_aux_data:
            # Extract and crop inverse variance
            ivar = self.crop(image_data["ivar"])

            # Extract and crop mask
            mask = image_data["mask"]
            if isinstance(mask, np.ndarray):
                mask = torch.as_tensor(mask, dtype=torch.float32)
            mask = self.crop(mask)
            if self.invert_mask:
                mask = 1 - mask

            # Apply augmentation to flux, ivar, and mask together
            if self.augment:
                flux, ivar, mask = random_rotation_and_flip(flux, ivar, mask)
        else:
            # Apply augmentation to flux only
            if self.augment:
                (flux,) = random_rotation_and_flip(flux)

        # Normalize flux after augmentation
        flux_normalized = self.normalize(flux)

        # Build return value based on parameters
        result = [flux_normalized]

        # Add auxiliary data if requested
        if self.return_aux_data:
            result.extend([ivar, mask])

        # Add conditioning variables if requested
        if self.condition_cols:
            cond = torch.tensor(
                [float(sample[c]) for c in self.condition_cols],
                dtype=torch.float32,
            )

            # Normalize conditioning if function provided
            if self.conditional_norm_fn is not None:
                cond = self.conditional_norm_fn(cond)

            result.append(cond)

        # Return single item or tuple
        if len(result) == 1:
            # when return_aux_data=False, condition_cols=None
            return result[0]
        return tuple(result)


def get_dataset_and_loaders(
    dataset_raw: Dataset,
    nx: int = 64,
    image_norm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    split: float = 0.8,
    batch_size: int = 128,
    num_workers: int = 8,
    invert_mask: bool = False,
    augment: bool = False,
) -> Tuple[HSCDataset, DataLoader, DataLoader]:
    dataset_raw = dataset_raw.select_columns(["image"]).with_format("torch")

    n_gals = len(dataset_raw)

    dataset = HSCDataset(
        dataset_raw,
        nx=nx,
        image_norm_fn=image_norm_fn,
        invert_mask=invert_mask,
        augment=augment,
    )
    n_bands, n_x, n_y = dataset[0][0].shape  # First element of tuple is flux
    print(f"Images dimension: {n_bands}*{n_x}*{n_y} ({n_gals} galaxies)")

    dataset_train, dataset_test = random_split(dataset, [split, 1 - split])

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return dataset, train_loader, test_loader
