"""Adapters for Spotify's pedalboard library."""

from typing import Any, Dict, Optional, Union

import numpy as np

from auementations.config.config_store import auementations_store
from auementations.core.base import BaseAugmentation
from auementations.core.parameters import ParameterSampler, ParameterValue


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
        sample_rate: int,
        p: float = 1.0,
        seed: Optional[int] = None,
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
        super().__init__(sample_rate=sample_rate, p=p, seed=seed)

        pedalboard = _lazy_import_pedalboard()
        self.pedalboard = pedalboard

        self.effect_class = effect_class
        self.param_specs = params
        self.current_params = {}

        # Initialize effect with resolved parameters
        self.randomize_parameters()
        self._init_effect()

    def _init_effect(self):
        """Initialize the underlying pedalboard effect."""
        self.effect = self.effect_class(**self.current_params)

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
                   Shape: (channels, samples) or (samples,)
            **kwargs: Additional parameters.

        Returns:
            Augmented audio in same format as input.
        """
        if not self.should_apply():
            return audio

        # Check if input is torch tensor and convert to numpy
        was_torch = False
        try:
            import torch

            was_torch = isinstance(audio, torch.Tensor)
            if was_torch:
                original_device = audio.device
                audio = audio.cpu().numpy()
        except ImportError:
            pass  # torch not available

        # Ensure we have numpy array
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Expected numpy array or torch.Tensor, got {type(audio)}")

        # Pedalboard expects float32 or float64
        original_dtype = audio.dtype
        if audio.dtype not in (np.float32, np.float64):
            audio = audio.astype(np.float32)

        # Ensure we have channel dimension
        original_shape = audio.shape
        if audio.ndim == 1:
            # (samples,) -> (1, samples)
            audio = audio.reshape(1, -1)

        # Reinitialize effect with new parameters
        self.randomize_parameters()
        self._init_effect()

        # Apply effect
        # Pedalboard process expects (num_channels, num_samples) and sample_rate
        augmented = self.effect.process(audio, sample_rate=self.sample_rate)

        # Restore original shape
        if len(original_shape) == 1:
            augmented = augmented.squeeze(0)

        # Restore original dtype if needed
        if augmented.dtype != original_dtype:
            augmented = augmented.astype(original_dtype)

        # Convert back to torch tensor if needed
        if was_torch:
            import torch

            augmented = torch.from_numpy(augmented).to(original_device)

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


@auementations_store(name="lpf", group="pedalboard")
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
        sample_rate: int,
        cutoff_freq: Optional[float] = None,
        min_cutoff_freq: Optional[float] = None,
        max_cutoff_freq: Optional[float] = None,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        pedalboard = _lazy_import_pedalboard()

        # Handle cutoff frequency parameter
        if cutoff_freq is not None:
            cutoff_hz = cutoff_freq
        elif min_cutoff_freq is not None and max_cutoff_freq is not None:
            cutoff_hz = (min_cutoff_freq, max_cutoff_freq)
        else:
            # Default range
            cutoff_hz = (150.0, 7500.0)

        super().__init__(
            effect_class=pedalboard.LowpassFilter,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            cutoff_frequency_hz=cutoff_hz,
        )


@auementations_store(name="hpf", group="pedalboard")
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
        sample_rate: int,
        cutoff_freq: Optional[float] = None,
        min_cutoff_freq: Optional[float] = None,
        max_cutoff_freq: Optional[float] = None,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        pedalboard = _lazy_import_pedalboard()

        # Handle cutoff frequency parameter
        if cutoff_freq is not None:
            cutoff_hz = cutoff_freq
        elif min_cutoff_freq is not None and max_cutoff_freq is not None:
            cutoff_hz = (min_cutoff_freq, max_cutoff_freq)
        else:
            # Default range
            cutoff_hz = (20.0, 2400.0)

        super().__init__(
            effect_class=pedalboard.HighpassFilter,
            sample_rate=sample_rate,
            p=p,
            seed=seed,
            cutoff_frequency_hz=cutoff_hz,
        )


__all__ = [
    "PedalboardAdapter",
    "LowPassFilter",
    "HighPassFilter",
]
