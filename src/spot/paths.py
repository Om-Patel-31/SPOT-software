"""Stable locations for SPOT configuration and persistent application data."""

from __future__ import annotations

import sys
from shutil import copy2, copytree
from pathlib import Path


def is_frozen() -> bool:
    """Return whether SPOT is running from its packaged executable."""
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    """Return the repository root while running from source."""
    return Path(__file__).resolve().parents[2]


def application_root() -> Path:
    """Return the folder holding the executable or source checkout."""
    return Path(sys.executable).resolve().parent if is_frozen() else project_root()


def models_dir() -> Path:
    """Return the writable identity/model store for this installation.

    A one-file executable unpacks bundled files into a temporary directory.
    Seed a persistent folder beside the executable on first launch so the face
    model and enrolled identities survive after the process exits.
    """
    directory = application_root() / "models" if is_frozen() else project_root() / "data" / "models"
    directory.mkdir(parents=True, exist_ok=True)
    bundle_directory = Path(getattr(sys, "_MEIPASS", "")) / "models"
    if is_frozen() and bundle_directory.is_dir():
        for source in bundle_directory.iterdir():
            destination = directory / source.name
            if destination.exists():
                continue
            if source.is_dir():
                copytree(source, destination)
            else:
                copy2(source, destination)
    return directory


def model_path(*parts: str) -> Path:
    """Return a path inside the writable shared model store."""
    return models_dir().joinpath(*parts)


def environment_file() -> Path:
    """Return the local, untracked environment file location."""
    return application_root() / ".env" if is_frozen() else project_root() / "config" / ".env"
