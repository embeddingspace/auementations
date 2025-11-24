"""Hydra configuration support for Auementations."""

from auementations.config import config_store
from auementations.config.config_store import auementations_store

# Import modules to trigger decorator registration
# This ensures all augmentation configs are registered in the store
from auementations.core import composition  # noqa: F401

__all__ = [
    "config_store",
    "auementations_store",
]
