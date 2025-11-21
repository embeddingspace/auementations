"""Hydra configuration support for Auementations."""

from auementations.config.store import auementations_store
from auementations.config.structured import (
    build_augmentation_config,
    register_auementations_configs,
)

__all__ = [
    "auementations_store",
    "build_augmentation_config",
    "register_auementations_configs",
]
