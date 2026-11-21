"""Local/Windows setup helpers for the PhenoMe tutorial notebook.

Import this module from the notebook; importing it does not change directories.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def find_phenome_repo_root(start: Path | None = None) -> Path:
    """Find the repository containing phenome/__init__.py."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "phenome" / "__init__.py").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find 'phenome/__init__.py' in {here} or its parent directories."
    )


def configure_windows_dll_paths() -> list:
    """Add common Conda/Python DLL folders on Windows and retain their handles."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []

    prefix = Path(sys.prefix)
    handles = []
    for folder in (prefix / "Library" / "bin", prefix / "DLLs", prefix):
        if folder.is_dir():
            handles.append(os.add_dll_directory(str(folder)))

    # Keep handles alive for the lifetime of the Python process.
    global _DLL_DIRECTORY_HANDLES
    _DLL_DIRECTORY_HANDLES = handles
    return handles


def setup_local_phenome(start: Path | None = None, verify_imports: bool = True) -> Path:
    """Add the PhenoMe repository to sys.path and optionally verify key imports."""
    repo_root = find_phenome_repo_root(start)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    configure_windows_dll_paths()

    if verify_imports:
        from PIL import Image  # noqa: F401
        from phenome import PhenoMe, load_dinov2_model, make_dataframe_metadata_fn  # noqa: F401

    return repo_root
