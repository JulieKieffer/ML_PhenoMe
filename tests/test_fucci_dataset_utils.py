"""Tests for the FUCCI dataset conversion helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Notebooks.tutorials.fucci_dataset_utils import prepare_fucci_subset


def test_prepare_fucci_subset_adds_time_aligned_labels(tmp_path: Path) -> None:
    """Each exported frame's metadata includes labels from the same timepoint."""
    raw_root = tmp_path / "raw"
    images_dir = raw_root / "train" / "images"
    labels_dir = raw_root / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    np.save(images_dir / "track.npy", np.zeros((3, 2, 4, 4), dtype=np.float32))
    labels = np.array([[[10], [11], [12]], [[20], [21], [22]]], dtype=np.float32)
    np.save(labels_dir / "track.npy", labels)

    _, metadata_path = prepare_fucci_subset(
        "train", raw_root=raw_root, out_root=tmp_path / "out", frame_stride=2
    )

    metadata = pd.read_csv(metadata_path)
    assert metadata["frame"].tolist() == [0, 2]
    assert metadata["fucci_label_0"].tolist() == [10.0, 12.0]
    assert metadata["fucci_label_1"].tolist() == [20.0, 22.0]
    assert metadata["fucci_label_mean"].tolist() == [15.0, 17.0]


def test_prepare_fucci_subset_rejects_short_label_track(tmp_path: Path) -> None:
    """Do not silently export frames that have no matching label timepoint."""
    raw_root = tmp_path / "raw"
    images_dir = raw_root / "train" / "images"
    labels_dir = raw_root / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    np.save(images_dir / "track.npy", np.zeros((3, 2, 4, 4), dtype=np.float32))
    np.save(labels_dir / "track.npy", np.zeros((2, 2, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="frame 2 was requested"):
        prepare_fucci_subset(
            "train", raw_root=raw_root, out_root=tmp_path / "out", frame_stride=2
        )
