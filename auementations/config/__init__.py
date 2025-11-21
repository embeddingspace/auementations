"""Hydra configuration support for Auementations."""

from auementations.config.structured import (
    build_augmentation_config,
    register_auementations_configs,
)

__all__ = [
    "build_augmentation_config",
    "register_auementations_configs",
]
