"""Tests for the FUCCI dataset conversion helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from Notebooks.tutorials.fucci_dataset_utils import prepare_fucci_subset


def _synthetic_track(num_frames: int = 3) -> np.ndarray:
    """Create channel-first frames with a bright, central nucleus."""
    track = np.zeros((num_frames, 2, 64, 64), dtype=np.float32)
    rows, cols = np.ogrid[:64, :64]
    nucleus = (rows - 32) ** 2 + (cols - 32) ** 2 <= 12**2
    track[:, 0] = 25
    track[:, 1, nucleus] = 200
    return track


def test_prepare_fucci_subset_adds_time_aligned_labels(tmp_path: Path) -> None:
    """Exported images and masks are cropped while labels stay time-aligned."""
    raw_root = tmp_path / "raw"
    images_dir = raw_root / "train" / "images"
    labels_dir = raw_root / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    np.save(images_dir / "track.npy", _synthetic_track())
    labels = np.array([[[10], [11], [12]], [[20], [21], [22]]], dtype=np.float32)
    np.save(labels_dir / "track.npy", labels)

    image_dir, metadata_path = prepare_fucci_subset(
        "train", raw_root=raw_root, out_root=tmp_path / "out", frame_stride=2
    )

    metadata = pd.read_csv(metadata_path)
    assert metadata["frame"].tolist() == [0, 2]
    assert metadata["fucci_label_0"].tolist() == [10.0, 12.0]
    assert metadata["fucci_label_1"].tolist() == [20.0, 22.0]
    assert metadata["fucci_label_mean"].tolist() == [15.0, 17.0]

    filename = metadata.loc[0, "filename"]
    exported_image = tifffile.imread(image_dir / filename)
    exported_mask = tifffile.imread(image_dir.parent / "FUCCI_masks" / filename)
    assert exported_image.shape[1:] == exported_mask.shape
    assert exported_mask.dtype == np.uint8
    assert np.any(exported_mask)
    assert exported_image.shape[1] < 64
    assert exported_image.shape[1] > 25
    assert not np.any(exported_mask[0, :])
    assert np.any(exported_image[:, exported_mask == 0] != 0)


def test_prepare_fucci_subset_rejects_short_label_track(tmp_path: Path) -> None:
    """Do not silently export frames that have no matching label timepoint."""
    raw_root = tmp_path / "raw"
    images_dir = raw_root / "train" / "images"
    labels_dir = raw_root / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    np.save(images_dir / "track.npy", _synthetic_track())
    np.save(labels_dir / "track.npy", np.zeros((2, 2, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="frame 2 was requested"):
        prepare_fucci_subset("train", raw_root=raw_root, out_root=tmp_path / "out", frame_stride=2)


def test_prepare_fucci_subset_skips_frames_without_nuclei(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    images_dir = raw_root / "train" / "images"
    images_dir.mkdir(parents=True)
    np.save(images_dir / "empty.npy", np.zeros((1, 2, 64, 64), dtype=np.float32))

    image_dir, metadata_path = prepare_fucci_subset(
        "train", raw_root=raw_root, out_root=tmp_path / "out"
    )

    metadata = pd.read_csv(metadata_path)
    assert metadata.empty
    assert not list(image_dir.rglob("*.tiff"))
    assert not list((image_dir.parent / "FUCCI_masks").rglob("*.tiff"))


def test_prepare_combined_subset_copies_masks(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    for subset in ("train", "test", "test_drugged"):
        images_dir = raw_root / subset / "images"
        images_dir.mkdir(parents=True)
        np.save(images_dir / "track.npy", _synthetic_track(num_frames=1))

    combined_image_dir, metadata_path = prepare_fucci_subset(
        "combined", raw_root=raw_root, out_root=tmp_path / "out"
    )

    metadata = pd.read_csv(metadata_path)
    assert len(metadata) == 3
    for filename in metadata["filename"]:
        mask_path = combined_image_dir.parent / "FUCCI_masks" / filename
        assert mask_path.is_file()
        assert np.any(tifffile.imread(mask_path))
