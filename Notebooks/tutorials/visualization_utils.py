
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt



def normalize_channel(image, percentiles=(1, 99)):
    """Normalize intensities for visualization without modifying source data."""
    image = np.asarray(image, dtype=np.float32)
    low, high = np.nanpercentile(image, percentiles)

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(image)

    return np.clip((image - low) / (high - low), 0, 1)


def display_dataset_image(
    data,
    dataset_name=None,
    title=None,
    percentiles=(1, 99),
    figsize=(13, 5),
):
    """
    Display common PhenoMe microscopy image formats.

    Supported formats:
    - E. coli: (H, W, 3), RGB
    - FUCCI: (2, H, W), two fluorescence channels
    - Grayscale: (H, W)
    """
    data = np.asarray(data)
    fig = None

    # --------------------------------------------------
    # FUCCI: (2, H, W)
    # --------------------------------------------------
    if dataset_name is not None and dataset_name.lower() == "fucci":
        if data.ndim != 3 or data.shape[0] != 2:
            raise ValueError(
                f"Expected FUCCI shape (2, H, W), got {data.shape}"
            )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        channel_0 = normalize_channel(data[0], percentiles)
        channel_1 = normalize_channel(data[1], percentiles)

        axes[0].imshow(channel_0, cmap="Reds", vmin=0, vmax=1)
        axes[0].set_title("Channel 0 — red")

        axes[1].imshow(channel_1, cmap="Greens", vmin=0, vmax=1)
        axes[1].set_title("Channel 1 — green")

        rgb = np.zeros((*data.shape[1:], 3), dtype=np.float32)
        rgb[..., 0] = channel_0
        rgb[..., 1] = channel_1

        axes[2].imshow(rgb)
        axes[2].set_title("Combined FUCCI channels")

        for ax in axes:
            ax.axis("off")

    # --------------------------------------------------
    # E. coli / RGB: (H, W, 3)
    # --------------------------------------------------
    elif data.ndim == 3 and data.shape[-1] == 3:
        data_float = data.astype(np.float32)
        low, high = np.percentile(data_float, percentiles)

        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            rgb = np.zeros_like(data_float)
        else:
            rgb = np.clip((data_float - low) / (high - low), 0, 1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(rgb)
        ax.set_title(title or "RGB microscopy image")
        ax.axis("off")

    # --------------------------------------------------
    # Grayscale: (H, W)
    # --------------------------------------------------
    elif data.ndim == 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(
            normalize_channel(data, percentiles),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        ax.set_title(title or "Grayscale microscopy image")
        ax.axis("off")

    else:
        raise ValueError(
            f"Unsupported image shape {data.shape}. "
            "Please specify its image layout."
        )

    if title and dataset_name and dataset_name.lower() == "fucci":
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()


'''


def normalize_channel(image, percentiles=(1, 99)):
    """Normalize intensities for visualization without modifying source data."""
    image = np.asarray(image, dtype=np.float32)
    low, high = np.nanpercentile(image, percentiles)

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(image)

    return np.clip((image - low) / (high - low), 0, 1)


def display_dataset_image(
    data,
    dataset_name=None,
    title=None,
    percentiles=(1, 99),
    figsize=(13, 5)
):
    """
    Display common PhenoMe microscopy image formats.

    Supported formats:
    - E. coli: (H, W, 3), RGB
    - FUCCI: (C, H, W), two fluorescence channels
    - Grayscale: (H, W)
    """

    data = np.asarray(data)

    fig = None

    # --------------------------------------------------
    # FUCCI: (2, H, W)
    # --------------------------------------------------
    if dataset_name is not None and dataset_name.lower() == "fucci":

        if data.ndim != 3 or data.shape[0] != 2:
            raise ValueError(
                f"Expected FUCCI shape (2, H, W), got {data.shape}"
            )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        channel_0 = normalize_channel(data[0], percentiles)
        channel_1 = normalize_channel(data[1], percentiles)

        axes[0].imshow(channel_0, cmap="Reds", vmin=0, vmax=1)
        axes[0].set_title("Channel 0 — red")

        axes[1].imshow(channel_1, cmap="Greens", vmin=0, vmax=1)
        axes[1].set_title("Channel 1 — green")

        rgb = np.zeros((*data.shape[1:], 3), dtype=np.float32)
        rgb[..., 0] = channel_0
        rgb[..., 1] = channel_1

        axes[2].imshow(rgb)
        axes[2].set_title("Combined FUCCI channels")

        for ax in axes:
            ax.axis("off")

    # --------------------------------------------------
    # E. coli: (H, W, 3)
    # --------------------------------------------------
    elif data.ndim == 3 and data.shape[-1] == 3:

        # Scale all RGB channels consistently, preserving colour ratios
        data_float = data.astype(np.float32)
        low, high = np.percentile(data_float, percentiles)

        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            rgb = np.zeros_like(data_float)
        else:
            rgb = np.clip((data_float - low) / (high - low), 0, 1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(rgb)
        ax.set_title(title or "RGB microscopy image")
        ax.axis("off")

    # --------------------------------------------------
    # Grayscale: (H, W)
    # --------------------------------------------------
    elif data.ndim == 2:

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(
            normalize_channel(data, percentiles),
            cmap="gray",
            vmin=0,
            vmax=1
        )
        ax.set_title(title or "Grayscale microscopy image")
        ax.axis("off")

    else:
        raise ValueError(
            f"Unsupported image shape {data.shape}. "
            "Please specify its image layout."
        )

    if title and dataset_name and dataset_name.lower() == "fucci":
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()


'''