"""Centralized hydra-zen store for Auementations augmentation configs.

This module provides a ZenStore that can be imported and used by external
repositories to access all Auementations augmentation configurations.

Example usage in external repository:
    from auementations.config import auementations_store
    from hydra_zen import ZenStore

    # Create your own store and merge Auementations configs
    my_store = ZenStore(name="my_project")
    auementations_store(my_store)

    # Now all Auementations augmentation configs are available
    # my_store.check_store("augmentation/composition", "compose")
"""

from hydra_zen import ZenStore, builds

from auementations.core.composition import Compose, OneOf, SomeOf


def _create_auementations_store() -> ZenStore:
    """Create and populate the Auementations store with all augmentation configs.

    Returns:
        ZenStore: Populated store with all Auementations augmentation configurations.
    """
    store = ZenStore(name="auementations")

    # Register composition configs
    _register_composition_configs(store)

    # Register torch_audiomentations adapter configs
    _register_torch_audiomentations_configs(store)

    return store


def _register_composition_configs(store: ZenStore) -> None:
    """Register composition augmentation configs to the store.

    Args:
        store: The ZenStore to register configs to.
    """
    # Create structured configs for composition classes
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

    # Register to store
    store(
        ComposeConfig,
        group="augmentation/composition",
        name="compose",
    )

    store(
        OneOfConfig,
        group="augmentation/composition",
        name="oneof",
    )

    store(
        SomeOfConfig,
        group="augmentation/composition",
        name="someof",
    )


def _register_torch_audiomentations_configs(store: ZenStore) -> None:
    """Register torch_audiomentations adapter configs to the store.

    Args:
        store: The ZenStore to register configs to.
    """
    # Import adapters - these handle lazy loading of torch dependencies
    from auementations.adapters.torch_audiomentations import (
        AddColoredNoise,
        Gain,
        HighPassFilter,
        LowPassFilter,
        PitchShift,
        TimeStretch,
    )

    # Create structured configs for each adapter
    GainConfig = builds(
        Gain,
        populate_full_signature=True,
        zen_partial=False,
    )

    PitchShiftConfig = builds(
        PitchShift,
        populate_full_signature=True,
        zen_partial=False,
    )

    AddColoredNoiseConfig = builds(
        AddColoredNoise,
        populate_full_signature=True,
        zen_partial=False,
    )

    HighPassFilterConfig = builds(
        HighPassFilter,
        populate_full_signature=True,
        zen_partial=False,
    )

    LowPassFilterConfig = builds(
        LowPassFilter,
        populate_full_signature=True,
        zen_partial=False,
    )

    TimeStretchConfig = builds(
        TimeStretch,
        populate_full_signature=True,
        zen_partial=False,
    )

    # Register to store with snake_case names
    store(
        GainConfig,
        group="augmentation/torch_audiomentations",
        name="gain",
    )

    store(
        PitchShiftConfig,
        group="augmentation/torch_audiomentations",
        name="pitch_shift",
    )

    store(
        AddColoredNoiseConfig,
        group="augmentation/torch_audiomentations",
        name="add_colored_noise",
    )

    store(
        HighPassFilterConfig,
        group="augmentation/torch_audiomentations",
        name="high_pass_filter",
    )

    store(
        LowPassFilterConfig,
        group="augmentation/torch_audiomentations",
        name="low_pass_filter",
    )

    store(
        TimeStretchConfig,
        group="augmentation/torch_audiomentations",
        name="time_stretch",
    )


# Create the singleton Auementations store
auementations_store = _create_auementations_store()


__all__ = ["auementations_store"]
