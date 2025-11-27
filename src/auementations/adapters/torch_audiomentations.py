"""Adapters for torch_audiomentations library."""

from typing import Any, Dict, Optional, Union

import numpy as np

from auementations.config.config_store import auementations_store
from auementations.core.base import BaseAugmentation
from auementations.core.parameters import ParameterSampler, ParameterValue


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
        super().__init__(sample_rate=sample_rate, p=p, seed=seed)

        torch, _ = _lazy_import_torch_audiomentations()
        self.torch = torch

        self.transform_class = transform_class
        self.param_specs = params
        self.current_params = {}

        # Initialize transform with resolved parameters
        self.randomize_parameters()
        self._init_transform()

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
                self.current_params[key] = ParameterSampler.sample(value)
            else:
                self.current_params[key] = value

    def __call__(
        self, audio: Union[np.ndarray, Any], **kwargs
    ) -> Union[np.ndarray, Any]:
        """Apply augmentation.

        Args:
            audio: Input audio as numpy array or torch tensor.
                   Shape: (batch, channels, samples) or (channels, samples) or (samples,)
            **kwargs: Additional parameters.

        Returns:
            Augmented audio in same format as input.
        """
        if not self.should_apply():
            return audio

        # Convert to torch tensor if needed
        was_numpy = isinstance(audio, np.ndarray)
        if was_numpy:
            audio_tensor = self.torch.from_numpy(audio).float()
        else:
            audio_tensor = audio

        # Ensure we have batch dimension
        original_shape = audio_tensor.shape
        if audio_tensor.ndim == 1:
            # (samples,) -> (1, 1, samples)
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        elif audio_tensor.ndim == 2:
            # (channels, samples) -> (1, channels, samples)
            audio_tensor = audio_tensor.unsqueeze(0)

        # Apply transform
        augmented = self.transform(audio_tensor, sample_rate=self.sample_rate)

        # Restore original shape
        if len(original_shape) == 1:
            augmented = augmented.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            augmented = augmented.squeeze(0)

        # Convert back to numpy if needed
        if was_numpy:
            return augmented.numpy()
        return augmented

    def to_config(self) -> Dict[str, Any]:
        """Export configuration."""
        config = super().to_config()
        config["transform_class"] = (
            f"{self.transform_class.__module__}.{self.transform_class.__name__}"
        )
        config.update(self.param_specs)
        return config


# Convenience wrappers for common torch_audiomentations transforms


@auementations_store(name="gain", group="torch_audiomentations")
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
    ):
        _, torch_audiomentations = _lazy_import_torch_audiomentations()
        super().__init__(
            transform_class=torch_audiomentations.Gain,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            min_gain_in_db=min_gain_db,
            max_gain_in_db=max_gain_db,
        )


@auementations_store(name="pitch_shift", group="torch_audiomentations")
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
    ):
        _, torch_audiomentations = _lazy_import_torch_audiomentations()
        super().__init__(
            transform_class=torch_audiomentations.PitchShift,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            min_transpose_semitones=min_semitones,
            max_transpose_semitones=max_semitones,
        )


@auementations_store(name="add_colored_noise", group="torch_audiomentations")
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
    ):
        _, torch_audiomentations = _lazy_import_torch_audiomentations()
        super().__init__(
            transform_class=torch_audiomentations.AddColoredNoise,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            min_snr_in_db=min_snr_db,
            max_snr_in_db=max_snr_db,
            min_f_decay=min_f_decay,
            max_f_decay=max_f_decay,
        )


@auementations_store(name="hpf", group="torch_audiomentations")
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
    ):
        _, torch_audiomentations = _lazy_import_torch_audiomentations()
        super().__init__(
            transform_class=torch_audiomentations.HighPassFilter,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
        )


@auementations_store(name="lpf", group="torch_audiomentations")
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
    ):
        _, torch_audiomentations = _lazy_import_torch_audiomentations()
        super().__init__(
            transform_class=torch_audiomentations.LowPassFilter,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
        )


@auementations_store(name="time_stretch", group="torch_audiomentations")
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
    ):
        _, torch_audiomentations = _lazy_import_torch_audiomentations()
        super().__init__(
            transform_class=torch_audiomentations.TimeStretch,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
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
