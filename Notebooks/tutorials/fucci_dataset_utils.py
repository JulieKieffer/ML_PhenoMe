"""FUCCI dataset inspection and conversion helpers for PhenoMe.

Raw input layout:
    fucci_datasets/<subset>/images/*.npy
    fucci_datasets/<subset>/labels/*.npy

Output layout:
    fucci_toy_datasets/<subset>/FUCCI_images/*.tiff
    fucci_toy_datasets/<subset>/metadata.csv

TIFF frames are written without intensity normalization or dtype conversion.
"""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt



TIME_STEP_MINUTES = 5
IMAGE_EXTENSIONS = {".tif", ".tiff"}

FUCCI_SUBSET_CONFIGS = {
    "train": {"label": "FUCCI — train", "condition": "train"},
    "test": {
        "label": "FUCCI — test (untreated / control)",
        "condition": "test",
    },
    "test_drugged": {
        "label": "FUCCI — test_drugged (drug-treated)",
        "condition": "drug_treated",
    },
}


def resolve_fucci_roots(start: Path | None = None) -> tuple[Path, Path]:
    """Find fucci_datasets above the current directory.

    If it is not found, return conventional sibling paths below the starting
    directory so the caller gets a useful, predictable location to inspect.
    """
    here = (start or Path.cwd()).resolve()
    for base in (here, *here.parents):
        raw_root = base / "fucci_datasets"
        if raw_root.is_dir():
            return raw_root, base / "fucci_toy_datasets"
    return here / "fucci_datasets", here / "fucci_toy_datasets"


def _npy_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".npy"
        and not p.name.startswith("._")
    )


def inspect_fucci_datasets(raw_root: Path | str | None = None,
                           subsets: Iterable[str] | None = None) -> dict:
    """Print and return image/label counts and sample array shapes."""
    if raw_root is None:
        raw_root, _ = resolve_fucci_roots()
    raw_root = Path(raw_root)
    subsets = list(subsets or FUCCI_SUBSET_CONFIGS)
    summary = {}

    for subset in subsets:
        subset_dir = raw_root / subset
        image_files = _npy_files(subset_dir / "images")
        label_files = _npy_files(subset_dir / "labels")
        info = {
            "image_count": len(image_files),
            "label_count": len(label_files),
            "sample_image": None,
            "sample_label": None,
        }
        if image_files:
            arr = np.load(image_files[0], allow_pickle=False, mmap_mode="r")
            info["sample_image"] = (image_files[0].name, arr.shape, arr.dtype)
        if label_files:
            arr = np.load(label_files[0], allow_pickle=False, mmap_mode="r")
            info["sample_label"] = (label_files[0].name, arr.shape, arr.dtype)

        summary[subset] = info
        print(f"=== {subset}: {subset_dir} ===")
        print(f"Images: {info['image_count']} | Labels: {info['label_count']}")
        if info["sample_image"]:
            print(f"Sample image: {info['sample_image']}")
        if info["sample_label"]:
            print(f"Sample label: {info['sample_label']}")
        if not subset_dir.is_dir():
            print("  [!] Subset folder not found.")
        print()

    return summary


def save_frame_as_tiff(frame: np.ndarray, output_path: Path | str) -> Path:
    """Write a frame as TIFF, preserving its values, dtype and array shape."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frame)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D frame, got shape {arr.shape}")
    # minisblack avoids treating a (C, H, W) microscopy array as an RGB image.
    tifffile.imwrite(output_path, arr, photometric="minisblack")
    return output_path


def _extract_track_frames(raw_arr: np.ndarray) -> list[np.ndarray]:
    """Return per-timepoint frames from a (T,C,H,W) track or a single image."""
    arr = np.asarray(raw_arr)
    if arr.ndim == 4:
        return [arr[t] for t in range(arr.shape[0])]
    return [arr]


def _clear_generated_images(folder: Path) -> None:
    """Remove old image outputs to prevent PNG/TIFF duplicates with same stem."""
    if folder.exists():
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | {".png"}:
                path.unlink()
    folder.mkdir(parents=True, exist_ok=True)


def prepare_fucci_subset(
    subset_key: str,
    raw_root: Path | str | None = None,
    out_root: Path | str | None = None,
    max_tracks: int | None = None,
    frame_stride: int = 1,
    clean_output: bool = True,
) -> tuple[Path, Path]:
    """Export a FUCCI subset (or 'combined') to TIFFs and metadata.csv.

    Set max_tracks=None to export every track. frame_stride=1 exports every
    frame; frame_stride=2 exports every second frame.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if raw_root is None or out_root is None:
        default_raw, default_out = resolve_fucci_roots()
        raw_root = Path(raw_root) if raw_root is not None else default_raw
        out_root = Path(out_root) if out_root is not None else default_out
    else:
        raw_root, out_root = Path(raw_root), Path(out_root)

    if subset_key == "combined":
        combined_dir = out_root / "combined"
        combined_img_dir = combined_dir / "FUCCI_images"
        if clean_output:
            _clear_generated_images(combined_img_dir)
        else:
            combined_img_dir.mkdir(parents=True, exist_ok=True)

        frames = []
        for sub_key in FUCCI_SUBSET_CONFIGS:
            sub_img_dir, sub_meta_path = prepare_fucci_subset(
                sub_key, raw_root, out_root, max_tracks, frame_stride, clean_output
            )
            sub_df = pd.read_csv(sub_meta_path)
            for row in sub_df.to_dict("records"):
                src = sub_img_dir / row["filename"]
                dst_name = f"{sub_key}_{row['filename']}"
                dst = combined_img_dir / dst_name
                if not src.is_file():
                    raise FileNotFoundError(f"Expected exported TIFF not found: {src}")
                shutil.copy2(src, dst)
                row["filename"] = dst_name
                row["dataset_split"] = sub_key
                frames.append(row)

        combined_meta = combined_dir / "metadata.csv"
        pd.DataFrame(frames).to_csv(combined_meta, index=False)
        print(f"✓ combined: {len(frames)} frames -> {combined_img_dir}")
        return combined_img_dir, combined_meta

    if subset_key not in FUCCI_SUBSET_CONFIGS:
        raise ValueError(
            f"Unknown subset {subset_key!r}. Choose one of {list(FUCCI_SUBSET_CONFIGS)} or 'combined'."
        )

    cfg = FUCCI_SUBSET_CONFIGS[subset_key]
    images_dir = raw_root / subset_key / "images"
    labels_dir = raw_root / subset_key / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"FUCCI images directory not found: {images_dir}")

    out_subset_dir = out_root / subset_key
    out_img_dir = out_subset_dir / "FUCCI_images"
    if clean_output:
        _clear_generated_images(out_img_dir)
    else:
        out_img_dir.mkdir(parents=True, exist_ok=True)

    npy_files = _npy_files(images_dir)
    if max_tracks is not None:
        npy_files = npy_files[:max_tracks]

    records = []
    for track_idx, npy_path in enumerate(npy_files):
        track = np.load(npy_path, allow_pickle=False)
        track_frames = _extract_track_frames(track)
        num_frames = len(track_frames)

        label_path = labels_dir / npy_path.name
        label_len = None
        if label_path.is_file():
            labels = np.load(label_path, allow_pickle=False)
            if labels.ndim >= 2:
                label_len = int(labels.shape[1])

        for t in range(0, num_frames, frame_stride):
            filename = f"{subset_key}_t{track_idx:04d}_f{t:04d}.tiff"
            save_frame_as_tiff(track_frames[t], out_img_dir / filename)
            records.append({
                "filename": filename,
                "source_file": npy_path.name,
                "track_id": npy_path.stem,
                "frame": t,
                "num_frames": num_frames,
                "time_min": t * TIME_STEP_MINUTES,
                "label_len": label_len,
                "dataset_split": subset_key,
                "condition": cfg["condition"],
            })

    out_subset_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_subset_dir / "metadata.csv"
    pd.DataFrame(records).to_csv(metadata_path, index=False)
    print(f"✓ {subset_key}: exported {len(records)} frames from {len(npy_files)} tracks -> {out_img_dir}")
    return out_img_dir, metadata_path


def display_tiff_conversion_example(
    frame: np.ndarray,
    output_path: Path | str,
    channel: int = 0,
    percentiles: tuple = (1, 99),
):
    """
    Save a frame as TIFF and display the original and converted images.

    Assumes 3D microscopy frames have shape (C, H, W).
    For 2D frames, the channel argument is ignored.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the comparison.
    reloaded : np.ndarray
        Image read back from the saved TIFF.
    """
    frame = np.asarray(frame)

    # Save the original frame as TIFF
    output_path = save_frame_as_tiff(frame, output_path)

    # Read the saved TIFF back
    reloaded = tifffile.imread(output_path)

    # Check whether the conversion preserved the array
    identical = (
        frame.shape == reloaded.shape
        and frame.dtype == reloaded.dtype
        and np.array_equal(frame, reloaded, equal_nan=True)
    )

    # Select the image to display
    if frame.ndim == 2:
        original_view = frame
        converted_view = reloaded

    elif frame.ndim == 3:
        if not 0 <= channel < frame.shape[0]:
            raise ValueError(
                f"Channel {channel} is out of range. "
                f"Available channels: 0 to {frame.shape[0] - 1}"
            )

        original_view = frame[channel]
        converted_view = reloaded[channel]

    else:
        raise ValueError(
            f"Expected a 2D or 3D frame, got shape {frame.shape}"
        )

    # Use shared contrast limits for a fair visual comparison.
    # This affects only how the images are displayed, not their data.
    vmin, vmax = np.nanpercentile(
        original_view, percentiles
    )

    if vmin == vmax:
        vmin = np.nanmin(original_view)
        vmax = np.nanmax(original_view)

    if vmin == vmax:
        vmax = vmin + 1

    # Display original and converted images
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(
        original_view,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Original NumPy frame")
    axes[0].axis("off")

    axes[1].imshow(
        converted_view,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Converted TIFF frame")
    axes[1].axis("off")

    fig.suptitle(
        f"TIFF conversion comparison — channel {channel}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()

    # Print conversion details
    print(f"Original shape:   {frame.shape}")
    print(f"TIFF shape:       {reloaded.shape}")
    print(f"Original dtype:   {frame.dtype}")
    print(f"TIFF dtype:       {reloaded.dtype}")
    print(f"Pixel values identical: {identical}")
    print(f"Saved to: {output_path}")

    return fig, reloaded

