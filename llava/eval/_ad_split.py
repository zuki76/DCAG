"""Shared splitting of six-view driving mosaics for training and evaluation."""

from __future__ import annotations

import torch


def should_split_ad_mosaic(
    image_path: str,
    image_folder: str = "",
    domain_name: str = "",
) -> bool:
    """Identify driving mosaics from the domain name or benchmark image path."""
    path_lc = str(image_path).lower()
    folder_lc = (image_folder or "").lower()
    return (
        "/drivelm/stitch/" in path_lc
        or ("stitch" in path_lc and path_lc.endswith(".jpg"))
        or folder_lc.endswith("/ad")
        or "/ad/" in folder_lc
        or domain_name == "AD"
    )


def apply_ad_split(image_tensor: torch.Tensor) -> torch.Tensor:
    """Split (C, H, W) into (6, C, H/2, W/3) views.

    Return the input unchanged if its shape cannot form a 2x3 grid.
    """
    if image_tensor.dim() != 3:
        return image_tensor
    height, width = image_tensor.shape[-2:]
    if height % 2 != 0 or width % 3 != 0:
        return image_tensor
    row_height = height // 2
    col_width = width // 3
    views = []
    for row in range(2):
        for col in range(3):
            views.append(
                image_tensor[
                    :,
                    row * row_height : (row + 1) * row_height,
                    col * col_width : (col + 1) * col_width,
                ]
            )
    return torch.stack(views, dim=0)
