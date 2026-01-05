"""Adapters for torch_audiomentations library."""

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from einops import rearrange

from auementations.config.config_store import auementations_store
from auementations.core.base import BaseAugmentation
from auementations.core.parameters import ParameterSampler


def _lazy_import_torch_audiomentations():
    """Lazy import to avoid requiring torch_audiomentations if not used."""
    try:
        import torch
        import torch_audiomentations

        return torch, torch_audiomentations
    except ImportError as e:
        raise ImportError(
            "torch_audiomentations backend requires 'torch' and 'torch_audiomentations'. "
            "Install with: pip install auementations[torch]"
        ) from e


class TorchAudiomentationsAdapter(BaseAugmentation):
    """Base adapter for torch_audiomentations transforms.

    This adapter wraps torch_audiomentations transforms to conform to the
    AUEM interface. It handles:
    - Parameter range sampling
    - Tensor/numpy conversions
    - Consistent sample_rate passing
    """

    def __init__(
        self,
        transform_class: Any,
        sample_rate: int | float,
        p: float = 1.0,
        seed: Optional[int] = None,
        mode: str = "per_example",
        **params,
    ):
        """Initialize adapter.

        Args:
            transform_class: torch_audiomentations transform class.
            sample_rate: Audio sample rate.
            p: Probability of applying augmentation.
            seed: Random seed.
            **params: Transform-specific parameters (can include ranges).
        """
        super().__init__(sample_rate=sample_rate, p=p, seed=seed, mode=mode)

        torch, _ = _lazy_import_torch_audiomentations()
        self.torch = torch

        self.transform_class = transform_class
        self.param_specs = params
        self.current_params = {}

        # Initialize transform with resolved parameters
        self.randomize_parameters()

    def _init_transform(self):
        """Initialize the underlying torch_audiomentations transform."""
        # torch_audiomentations uses 'sample_rate', not 'sr'
        self.transform = self.transform_class(
            sample_rate=self.sample_rate,
            p=1.0,  # We handle probability at our level
            **self.current_params,
        )

    def randomize_parameters(self) -> None:
        """Sample random parameters from configured ranges."""
        self.current_params = {}
        for key, value in self.param_specs.items():
            # Check if this is a range parameter
            if isinstance(value, (tuple, dict)):
                self.current_params[key] = ParameterSampler.sample(value, self.rng)
            else:
                self.current_params[key] = value

        self._init_transform()
        return self.current_params

    def to_config(self) -> Dict[str, Any]:
        """Export configuration."""
        config = super().to_config()
        config["transform_class"] = (
            f"{self.transform_class.__module__}.{self.transform_class.__name__}"
        )
        config.update(self.param_specs)
        return config

    def forward(self, audio: Tensor) -> Tensor:
        audio = torch.as_tensor(audio)

        match audio.ndim:
            case 1:
                audio = rearrange(audio, "t -> 1 1 t")
                augmented = self.transform(audio)
                augmented = rearrange(augmented, "1 1 t -> t")
            case 2:
                audio = rearrange(audio, "b t -> b 1 t")
                augmented = self.transform(audio)
                augmented = rearrange(audio, "b 1 t -> b t")
            case _:
                augmented = self.transform(audio)

        return augmented


# Convenience wrappers for common torch_audiomentations transforms


@auementations_store(name="gain", group="auementations/torch_audiomentations")
class Gain(TorchAudiomentationsAdapter):
    """Apply gain to audio.

    Args:
        sample_rate: Audio sample rate.
        min_gain_db: Minimum gain in dB (can be negative for attenuation).
        max_gain_db: Maximum gain in dB.
        p: Probability of applying.
    """

    def __init__(
        self,
        sample_rate: int | float,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        p: float = 1.0,
        seed: Optional[int] = None,
        mode: str = "per_example",
    ):
        _, torch_audiomentations = _lazy_import_torch_audiomentations()
        super().__init__(
            transform_class=torch_audiomentations.Gain,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            mode=mode,
            min_gain_in_db=min_gain_db,
            max_gain_in_db=max_gain_db,
        )


@auementations_store(name="pitch_shift", group="auementations/torch_audiomentations")
class PitchShift(TorchAudiomentationsAdapter):
    """Shift pitch of audio.

    Args:
        sample_rate: Audio sample rate.
        min_semitones: Minimum pitch shift in semitones (negative = down).
        max_semitones: Maximum pitch shift in semitones (positive = up).
        p: Probability of applying.
    """

    def __init__(
        self,
        sample_rate: int | float,
        min_semitones: float = -4.0,
        max_semitones: float = 4.0,
        p: float = 1.0,
        seed: Optional[int] = None,
        mode: str = "per_example",
    ):
        _, torch_audiomentations = _lazy_import_torch_audiomentations()
        super().__init__(
            transform_class=torch_audiomentations.PitchShift,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            mode=mode,
            min_transpose_semitones=min_semitones,
            max_transpose_semitones=max_semitones,
        )


@auementations_store(
    name="add_colored_noise", group="auementations/torch_audiomentations"
)
class AddColoredNoise(TorchAudiomentationsAdapter):
    """Add colored noise to audio.

    Args:
        sample_rate: Audio sample rate.
        min_snr_db: Minimum signal-to-noise ratio in dB.
        max_snr_db: Maximum signal-to-noise ratio in dB.
        min_f_decay: Minimum frequency decay factor (0=white, -2=brown, 2=blue).
        max_f_decay: Maximum frequency decay factor.
        p: Probability of applying.
    """

    def __init__(
        self,
        sample_rate: int | float,
        min_snr_db: float = 3.0,
        max_snr_db: float = 30.0,
        min_f_decay: float = -2.0,
        max_f_decay: float = 2.0,
        p: float = 1.0,
        seed: Optional[int] = None,
        mode: str = "per_example",
    ):
        _, torch_audiomentations = _lazy_import_torch_audiomentations()
        super().__init__(
            transform_class=torch_audiomentations.AddColoredNoise,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            mode=mode,
            min_snr_in_db=min_snr_db,
            max_snr_in_db=max_snr_db,
            min_f_decay=min_f_decay,
            max_f_decay=max_f_decay,
        )


@auementations_store(name="hpf", group="auementations/torch_audiomentations")
class HighPassFilter(TorchAudiomentationsAdapter):
    """Apply high-pass filter.

    Args:
        sample_rate: Audio sample rate.
        min_cutoff_freq: Minimum cutoff frequency in Hz.
        max_cutoff_freq: Maximum cutoff frequency in Hz.
        p: Probability of applying.
    """

    def __init__(
        self,
        sample_rate: int | float,
        min_cutoff_freq: float = 20.0,
        max_cutoff_freq: float = 2400.0,
        p: float = 1.0,
        seed: Optional[int] = None,
        mode: str = "per_example",
    ):
        _, torch_audiomentations = _lazy_import_torch_audiomentations()
        super().__init__(
            transform_class=torch_audiomentations.HighPassFilter,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            mode=mode,
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
        )


@auementations_store(name="lpf", group="auementations/torch_audiomentations")
class LowPassFilter(TorchAudiomentationsAdapter):
    """Apply low-pass filter.

    Args:
        sample_rate: Audio sample rate.
        min_cutoff_freq: Minimum cutoff frequency in Hz.
        max_cutoff_freq: Maximum cutoff frequency in Hz.
        p: Probability of applying.
    """

    def __init__(
        self,
        sample_rate: int | float,
        min_cutoff_freq: float = 150.0,
        max_cutoff_freq: float = 7500.0,
        p: float = 1.0,
        seed: Optional[int] = None,
        mode: str = "per_example",
    ):
        _, torch_audiomentations = _lazy_import_torch_audiomentations()
        super().__init__(
            transform_class=torch_audiomentations.LowPassFilter,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            mode=mode,
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
        )


@auementations_store(name="time_stretch", group="auementations/torch_audiomentations")
class TimeStretch(TorchAudiomentationsAdapter):
    """Stretch audio in time without changing pitch.

    Args:
        sample_rate: Audio sample rate.
        min_rate: Minimum stretch rate (<1.0 = slower, >1.0 = faster).
        max_rate: Maximum stretch rate.
        p: Probability of applying.
    """

    def __init__(
        self,
        sample_rate: int | float,
        min_rate: float = 0.8,
        max_rate: float = 1.25,
        p: float = 1.0,
        seed: Optional[int] = None,
        mode: str = "per_example",
    ):
        _, torch_audiomentations = _lazy_import_torch_audiomentations()
        super().__init__(
            transform_class=torch_audiomentations.TimeStretch,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            mode=mode,
            min_rate=min_rate,
            max_rate=max_rate,
        )


__all__ = [
    "Gain",
    "PitchShift",
    "AddColoredNoise",
    "HighPassFilter",
    "LowPassFilter",
    "TimeStretch",
]
