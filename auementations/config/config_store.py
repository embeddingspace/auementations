"""Centralized config store and registration decorator for Auementations.

Example usage:
    from auementations.config.config_store import auementations_store

    @auementations_store(name="Gain", group="augmentation/torch_audiomentations")
    class Gain:
        def __init__(self, sample_rate: int, gain_db: float):
            ...
"""

from hydra_zen import ZenStore
from hydra_zen.third_party.beartype import (
    validates_with_beartype,
)

# Create the singleton store
auementations_store = ZenStore(name="auementations")(
    populate_full_signature=True,
    hydra_convert="all",
    zen_wrappers=validates_with_beartype,
)


__all__ = ["auementations_store"]

from auementations.core.composition import *  # noqa
from auementations.adapters.torch_audiomentations import *  # noqa
