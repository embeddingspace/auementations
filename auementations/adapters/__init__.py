"""Adapter modules for different audio augmentation backends."""

from auementations.adapters.torch_audiomentations import (
    AddColoredNoise,
    Gain,
    HighPassFilter as TorchHighPassFilter,
    LowPassFilter as TorchLowPassFilter,
    PitchShift,
    TimeStretch,
    TorchAudiomentationsAdapter,
)

# Pedalboard adapters - may not be available if pedalboard not installed
try:
    from auementations.adapters.pedalboard import (
        HighPassFilter as PedalboardHighPassFilter,
        LowPassFilter as PedalboardLowPassFilter,
        PedalboardAdapter,
    )

    _PEDALBOARD_AVAILABLE = True
except ImportError:
    _PEDALBOARD_AVAILABLE = False
    PedalboardAdapter = None
    PedalboardHighPassFilter = None
    PedalboardLowPassFilter = None

__all__ = [
    # Torch audiomentations
    "TorchAudiomentationsAdapter",
    "Gain",
    "PitchShift",
    "AddColoredNoise",
    "TorchHighPassFilter",
    "TorchLowPassFilter",
    "TimeStretch",
    # Pedalboard
    "PedalboardAdapter",
    "PedalboardHighPassFilter",
    "PedalboardLowPassFilter",
]
