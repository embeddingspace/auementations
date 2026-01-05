"""Adapters for Spotify's pedalboard library."""

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor
from einops import rearrange

from auementations.config.config_store import auementations_store
from auementations.core.base import BaseAugmentation
from auementations.core.parameters import ParameterSampler


def _lazy_import_pedalboard():
    """Lazy import to avoid requiring pedalboard if not used."""
    try:
        import pedalboard

        return pedalboard
    except ImportError as e:
        raise ImportError(
            "pedalboard backend requires 'pedalboard'. "
            "Install with: pip install auementations[pedalboard]"
        ) from e


class PedalboardAdapter(BaseAugmentation):
    """Base adapter for pedalboard effects.

    This adapter wraps pedalboard effects to conform to the
    AUEM interface. It handles:
    - Parameter range sampling
    - Numpy array processing
    - Consistent sample_rate passing
    """

    def __init__(
        self,
        effect_class: Any,
        sample_rate: int | float,
        p: float = 1.0,
        seed: Optional[int] = None,
        mode: str = "per_example",
        **params,
    ):
        """Initialize adapter.

        Args:
            effect_class: pedalboard effect class.
            sample_rate: Audio sample rate.
            p: Probability of applying augmentation.
            seed: Random seed.
            **params: Effect-specific parameters (can include ranges).
        """
        super().__init__(sample_rate=sample_rate, p=p, mode=mode, seed=seed)

        pedalboard = _lazy_import_pedalboard()
        self.pedalboard = pedalboard

        self.effect_class = effect_class
        self.param_specs = params
        self.current_params = {}

        # Initialize effect with resolved parameters
        self.randomize_parameters()

    def _init_effect(self):
        """Initialize the underlying pedalboard effect."""
        self.effect = self.effect_class(**self.current_params)

    def randomize_parameters(self) -> None:
        """Sample random parameters from configured ranges."""
        self.current_params = {}
        for key, value in self.param_specs.items():
            # Check if this is a range parameter
            if isinstance(value, (tuple, dict)):
                self.current_params[key] = ParameterSampler.sample(value, self.rng)
            else:
                self.current_params[key] = value

        self._init_effect()
        return self.current_params

    def forward(self, audio: Tensor) -> Tensor:
        # Check if input is torch tensor and convert to numpy
        # -- pedalboard must operate on numpy arrays
        # of ndim 1 or 2.
        was_torch = isinstance(audio, torch.Tensor)
        if was_torch:
            original_device = audio.device
            audio = audio.cpu().numpy(force=True)

        # Pedalboard expects float32 or float64
        original_dtype = audio.dtype
        if audio.dtype not in (np.float32, np.float64):
            audio = audio.astype(np.float32)

        # Process based on the item's dimensionality
        # After slicing batch dimension, we have either:
        match audio.ndim:
            case 4:  # (batch, source, channel, samples)
                b, s, c, t = audio.shape
                audio_flat = rearrange(audio, "b s c t -> (b s c) t")
                augmented = self.effect.process(
                    audio_flat, sample_rate=self.sample_rate
                )
                augmented = rearrange(augmented, "(b s c) t -> b s c t", b=b, s=s, c=c)
            case 3:  # (source, channel, samples)
                # Flatten source and channel dimensions
                s, c, t = audio.shape
                audio_flat = rearrange(audio, "s c t -> (s c) t")
                augmented = self.effect.process(
                    audio_flat, sample_rate=self.sample_rate
                )
                augmented = rearrange(augmented, "(s c) t -> s c t", s=s, c=c)
            case _:  # 2D: (channels, samples)
                # Apply directly
                augmented = self.effect.process(audio, sample_rate=self.sample_rate)

        # Restore original dtype if needed
        if augmented.dtype != original_dtype:
            augmented = augmented.astype(original_dtype)

        # Convert back to torch tensor if needed
        if was_torch:
            augmented = torch.from_numpy(augmented).to(original_device)
            # Replace NaN and inf values with valid numbers
            augmented = torch.nan_to_num(augmented, nan=0.0, posinf=1.0, neginf=-1.0)

        return augmented

    def to_config(self) -> Dict[str, Any]:
        """Export configuration."""
        config = super().to_config()
        config["effect_class"] = (
            f"{self.effect_class.__module__}.{self.effect_class.__name__}"
        )
        config.update(self.param_specs)
        return config


# Convenience wrappers for common pedalboard effects


@auementations_store(name="lpf", group="auementations/pedalboard")
class LowPassFilter(PedalboardAdapter):
    """Apply low-pass filter.

    Args:
        sample_rate: Audio sample rate.
        cutoff_freq: Cutoff frequency in Hz (or use min/max for range).
        min_cutoff_freq: Minimum cutoff frequency in Hz (alternative to cutoff_freq).
        max_cutoff_freq: Maximum cutoff frequency in Hz (alternative to cutoff_freq).
        p: Probability of applying.
    """

    def __init__(
        self,
        sample_rate: int | float,
        cutoff_freq: Optional[float] = None,
        min_cutoff_freq: Optional[float] = None,
        max_cutoff_freq: Optional[float] = None,
        p: float = 1.0,
        seed: Optional[int] = None,
        mode: str = "per_example",
    ):
        pedalboard = _lazy_import_pedalboard()

        # Handle cutoff frequency parameter
        if cutoff_freq is not None:
            cutoff_hz = min(cutoff_freq, sample_rate / 2)
        elif min_cutoff_freq is not None and max_cutoff_freq is not None:
            cutoff_hz = (min_cutoff_freq, min(max_cutoff_freq, sample_rate / 2))
        else:
            # Default range
            cutoff_hz = (150.0, 7500.0)

        super().__init__(
            effect_class=pedalboard.LowpassFilter,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            mode=mode,
            cutoff_frequency_hz=cutoff_hz,
        )


@auementations_store(name="hpf", group="auementations/pedalboard")
class HighPassFilter(PedalboardAdapter):
    """Apply high-pass filter.

    Args:
        sample_rate: Audio sample rate.
        cutoff_freq: Cutoff frequency in Hz (or use min/max for range).
        min_cutoff_freq: Minimum cutoff frequency in Hz (alternative to cutoff_freq).
        max_cutoff_freq: Maximum cutoff frequency in Hz (alternative to cutoff_freq).
        p: Probability of applying.
    """

    def __init__(
        self,
        sample_rate: int | float,
        cutoff_freq: Optional[float] = None,
        min_cutoff_freq: Optional[float] = None,
        max_cutoff_freq: Optional[float] = None,
        p: float = 1.0,
        seed: Optional[int] = None,
        mode: str = "per_example",
    ):
        pedalboard = _lazy_import_pedalboard()

        # Handle cutoff frequency parameter
        if cutoff_freq is not None:
            cutoff_hz = min(cutoff_freq, sample_rate / 2)
        elif min_cutoff_freq is not None and max_cutoff_freq is not None:
            cutoff_hz = (min_cutoff_freq, min(max_cutoff_freq, sample_rate / 2))
        else:
            # Default range
            cutoff_hz = (20.0, 2400.0)

        super().__init__(
            effect_class=pedalboard.HighpassFilter,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            mode=mode,
            cutoff_frequency_hz=cutoff_hz,
        )


@auementations_store(name="peak_filter", group="auementations/pedalboard")
class PeakFilter(PedalboardAdapter):
    """Apply a parametric EQ peak filter (biquad) to audio.

    This augmentation applies a peak filter (also known as a bell filter or peaking EQ)
    to boost or cut a specific frequency range. It's commonly used for EQ adjustments
    in audio production.

    Args:
        sample_rate: Audio sample rate in Hz.
        min_center_freq: Minimum center frequency in Hz.
        max_center_freq: Maximum center frequency in Hz.
        min_gain_db: Minimum gain in dB (negative for cut, positive for boost).
        max_gain_db: Maximum gain in dB.
        min_q: Minimum Q factor (bandwidth). Higher Q = narrower filter.
        max_q: Maximum Q factor.
        p: Probability of applying the augmentation.
        mode: How to apply parameters across dimensions:
            - "per_batch": Same parameters for entire batch
            - "per_example": Different parameters per batch element (default)
            - "per_source": Different parameters per source (dim 1)
            - "per_channel": Different parameters per channel (dim 2)
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        sample_rate: int | float,
        min_center_freq: float = 200.0,
        max_center_freq: float = 8000.0,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        min_q: float = 0.707,
        max_q: float = 2.0,
        p: float = 1.0,
        mode: str = "per_example",
        seed: Optional[int] = None,
    ):
        pedalboard = _lazy_import_pedalboard()

        center_freq_hz = (min_center_freq, max_center_freq)
        gain_range_db = (min_gain_db, max_gain_db)
        q_range = (min_q, max_q)

        super().__init__(
            effect_class=pedalboard.PeakFilter,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            mode=mode,
            cutoff_frequency_hz=center_freq_hz,
            gain_db=gain_range_db,
            q=q_range,
        )


__all__ = [
    "PedalboardAdapter",
    "LowPassFilter",
    "HighPassFilter",
    "PeakFilter",
]
