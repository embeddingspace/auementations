"""Adapter modules for different audio augmentation backends."""

from auementations.adapters.torch_audiomentations import (
    AddColoredNoise,
    PitchShift,
    TimeStretch,
    TorchAudiomentationsAdapter,
)
from auementations.adapters.torch_audiomentations import (
    HighPassFilter as TorchHighPassFilter,
)
from auementations.adapters.torch_audiomentations import (
    LowPassFilter as TorchLowPassFilter,
)

# Pedalboard adapters - may not be available if pedalboard not installed
try:
    from auementations.adapters.pedalboard import (
        HighPassFilter as PedalboardHighPassFilter,
    )
    from auementations.adapters.pedalboard import (
        LowPassFilter as PedalboardLowPassFilter,
    )
    from auementations.adapters.pedalboard import (
        PedalboardAdapter,
    )

    _PEDALBOARD_AVAILABLE = True
except ImportError:
    _PEDALBOARD_AVAILABLE = False
    PedalboardAdapter = None
    PedalboardHighPassFilter = None
    PedalboardLowPassFilter = None

from auementations.adapters.custom import GainAugmentation

__all__ = [
    # Torch audiomentations
    "TorchAudiomentationsAdapter",
    "PitchShift",
    "AddColoredNoise",
    "TorchHighPassFilter",
    "TorchLowPassFilter",
    "TimeStretch",
    # Pedalboard
    "PedalboardAdapter",
    "PedalboardHighPassFilter",
    "PedalboardLowPassFilter",
    # auementations
    "GainAugmentation",
]
