"""Structured configs for Hydra integration using hydra-zen."""

from typing import Any, Dict, List, Optional, Union
from hydra_zen import builds, make_config
from dataclasses import dataclass

# Import core classes for building structured configs
from auementations.core.composition import Compose, OneOf, SomeOf


def build_augmentation_config(augmentation_cls: type, **kwargs) -> Any:
    """Build a structured config for an augmentation class.

    This is a convenience function for creating Hydra-compatible configs
    for any augmentation class.

    Args:
        augmentation_cls: The augmentation class to build config for.
        **kwargs: Default parameter values for the config.

    Returns:
        A hydra-zen builds object that can be used with Hydra.

    Example:
        >>> from auementations.config import build_augmentation_config
        >>> from auementations.adapters.torch_audiomentations import Gain
        >>> GainConfig = build_augmentation_config(
        ...     Gain,
        ...     sample_rate=16000,
        ...     min_gain_db=-12.0,
        ...     max_gain_db=12.0,
        ...     p=0.5
        ... )
    """
    return builds(augmentation_cls, **kwargs, zen_partial=False)


# Composition configs
ComposeConfig = builds(
    Compose,
    populate_full_signature=True,
    zen_partial=False,
)

OneOfConfig = builds(
    OneOf,
    populate_full_signature=True,
    zen_partial=False,
)

SomeOfConfig = builds(
    SomeOf,
    populate_full_signature=True,
    zen_partial=False,
)


def register_auementations_configs(config_store: Any, group: str = "augmentation") -> None:
    """Register Auementations configs with Hydra's ConfigStore.

    This function registers all composition configs so they can be used
    in Hydra config files.

    Args:
        config_store: Hydra's ConfigStore instance.
        group: Config group name to register under.

    Example:
        >>> from hydra import initialize, compose
        >>> from hydra.core.config_store import ConfigStore
        >>> from auementations.config import register_auementations_configs
        >>>
        >>> cs = ConfigStore.instance()
        >>> register_auementations_configs(cs)
    """
    config_store.store(group=group, name="compose", node=ComposeConfig)
    config_store.store(group=group, name="one_of", node=OneOfConfig)
    config_store.store(group=group, name="some_of", node=SomeOfConfig)


@dataclass
class AugmentationPipelineConfig:
    """Base config for augmentation pipelines.

    This can be used as a base for creating structured configs for
    entire augmentation pipelines in training configs.
    """
    sample_rate: int
    augmentations: Optional[List[Any]] = None


# Create a config builder for the pipeline
AugmentationPipelineConfigBuilder = make_config(
    sample_rate=int,
    augmentations=None,
)
