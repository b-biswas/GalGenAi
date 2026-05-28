"""Data augmentation utilities for galaxy images."""

import torch


def random_rotation_and_flip(*tensors):
    """Random 90° rotation and flips to multiple tensors simultaneously.

    Parameters:
    -----------
    *tensors: Variable number of tensors of shape (C, H, W)
        same rotation/slipping is applied to each.

    Returns:
    --------
    Tuple of augmented tensors in the same order as input
    """
    k = torch.randint(0, 4, (1,)).item()  # Rotation: (0°, 90°, 180°, 270°)
    flip_h = torch.rand(1).item() > 0.5
    flip_v = torch.rand(1).item() > 0.5

    def apply_transform(img):
        if k > 0:
            img = torch.rot90(img, k=k, dims=(1, 2))
        if flip_h:
            img = torch.flip(img, dims=(2,))
        if flip_v:
            img = torch.flip(img, dims=(1,))
        return img

    return tuple(apply_transform(t) for t in tensors)
