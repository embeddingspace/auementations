"""Adapters for Spotify's pedalboard library."""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
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
                   or (batch, channels, samples) or (batch, source, channel, samples)
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

        original_shape = audio.shape

        # Handle batched inputs (3D or 4D) - apply different parameters per example
        if audio.ndim >= 3:
            # Process each batch example with different parameters
            batch_size = audio.shape[0]
            augmented_list = []

            for i in range(batch_size):
                # Extract single batch item
                batch_item = audio[i]

                # Randomize parameters for this example
                self.randomize_parameters()
                self._init_effect()

                # Process based on the item's dimensionality
                # After slicing batch dimension, we have either:
                # - 2D: (channels, samples) for 3D input
                # - 3D: (source, channel, samples) for 4D input
                if batch_item.ndim == 3:  # (source, channel, samples)
                    # Flatten source and channel dimensions
                    s, c, t = batch_item.shape
                    batch_item_flat = rearrange(batch_item, "s c t -> (s c) t")
                    augmented_item = self.effect.process(
                        batch_item_flat, sample_rate=self.sample_rate
                    )
                    augmented_item = rearrange(
                        augmented_item, "(s c) t -> s c t", s=s, c=c
                    )
                else:  # 2D: (channels, samples)
                    # Apply directly
                    augmented_item = self.effect.process(
                        batch_item, sample_rate=self.sample_rate
                    )

                augmented_list.append(augmented_item)

            # Stack batch back together
            augmented = np.stack(augmented_list, axis=0)
        else:
            # Non-batched input (1D or 2D)
            # Ensure we have channel dimension for 1D input
            if audio.ndim == 1:
                audio = audio[None, :]

            # Randomize parameters
            self.randomize_parameters()
            self._init_effect()

            # Apply effect
            augmented = self.effect.process(audio, sample_rate=self.sample_rate)

            # Restore original shape for 1D
            if len(original_shape) == 1:
                augmented = augmented.squeeze(0)

        # Restore original dtype if needed
        if augmented.dtype != original_dtype:
            augmented = augmented.astype(original_dtype)

        # Convert back to torch tensor if needed
        if was_torch:
            import torch

            augmented = torch.from_numpy(augmented).to(original_device)
            # Replace NaN and inf values with valid numbers
            augmented = torch.nan_to_num(augmented, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            # If numpy, use numpy's nan_to_num
            augmented = np.nan_to_num(augmented, nan=0.0, posinf=1.0, neginf=-1.0)

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
            cutoff_frequency_hz=cutoff_hz,
        )


@auementations_store(name="peak_filter", group="auementations/pedalboard")
class PeakFilter(nn.Module):
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

    VALID_MODES = ["per_batch", "per_example", "per_source", "per_channel"]

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
        super().__init__()
        self.sample_rate = sample_rate
        self.min_center_freq = min_center_freq
        self.max_center_freq = max_center_freq
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.min_q = min_q
        self.max_q = max_q
        self.p = p
        self.seed = seed

        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.mode = mode

        self.generator = torch.Generator()
        if self.seed is not None:
            self.generator.manual_seed(self.seed)

        self.pedalboard = _lazy_import_pedalboard()
        self.current_params = {}

    def randomize_parameters(self) -> None:
        """Sample random parameters for the filter."""
        # Sample center frequency
        center_freq = (
            torch.rand(1, generator=self.generator).item()
            * (self.max_center_freq - self.min_center_freq)
            + self.min_center_freq
        )

        # Sample gain
        gain_db = (
            torch.rand(1, generator=self.generator).item()
            * (self.max_gain_db - self.min_gain_db)
            + self.min_gain_db
        )

        # Sample Q factor
        q = (
            torch.rand(1, generator=self.generator).item() * (self.max_q - self.min_q)
            + self.min_q
        )

        # Ensure center frequency doesn't exceed Nyquist
        center_freq = min(center_freq, self.sample_rate / 2 * 0.99)

        self.current_params = {
            "center_freq": center_freq,
            "gain_db": gain_db,
            "q": q,
        }

    def _apply_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply the peak filter to audio.

        Args:
            audio: Audio array with shape (channels, samples) or (samples,).

        Returns:
            Filtered audio with same shape.
        """
        # Ensure we have channel dimension
        original_shape = audio.shape
        if audio.ndim == 1:
            audio = audio[None, :]

        # Create and apply filter
        peak_filter = self.pedalboard.PeakFilter(
            cutoff_frequency_hz=self.current_params["center_freq"],
            gain_db=self.current_params["gain_db"],
            q=self.current_params["q"],
        )

        filtered = peak_filter.process(audio, sample_rate=self.sample_rate)

        # Restore original shape
        if len(original_shape) == 1:
            filtered = filtered.squeeze(0)

        return filtered

    def forward(
        self, x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Apply peak filter augmentation.

        Args:
            x: Input audio with shape (batch, source, channel, time).
               Also accepts numpy arrays.

        Returns:
            Augmented audio with same shape and type as input.
        """
        # Check probability
        p_apply = torch.rand((), generator=self.generator)
        if p_apply > self.p:
            return x

        # Track input type for output conversion
        was_torch = isinstance(x, torch.Tensor)
        if was_torch:
            original_device = x.device
            x_np = x.cpu().numpy()
        else:
            x_np = x

        # Ensure float32
        original_dtype = x_np.dtype
        if x_np.dtype not in (np.float32, np.float64):
            x_np = x_np.astype(np.float32)

        # Expected shape: (batch, source, channel, time)
        if x_np.ndim != 4:
            raise ValueError(
                f"Expected 4D input (batch, source, channel, time), got {x_np.ndim}D"
            )

        batch_size, n_sources, n_channels, n_samples = x_np.shape
        output = np.zeros_like(x_np)

        # Apply filter based on mode
        if self.mode == "per_batch":
            # Single set of parameters for entire batch
            self.randomize_parameters()
            for b in range(batch_size):
                for s in range(n_sources):
                    for c in range(n_channels):
                        output[b, s, c, :] = self._apply_filter(x_np[b, s, c, :])

        elif self.mode == "per_example":
            # Different parameters per batch example
            for b in range(batch_size):
                self.randomize_parameters()
                for s in range(n_sources):
                    for c in range(n_channels):
                        output[b, s, c, :] = self._apply_filter(x_np[b, s, c, :])

        elif self.mode == "per_source":
            # Different parameters per source
            for b in range(batch_size):
                for s in range(n_sources):
                    self.randomize_parameters()
                    for c in range(n_channels):
                        output[b, s, c, :] = self._apply_filter(x_np[b, s, c, :])

        elif self.mode == "per_channel":
            # Different parameters per channel
            for b in range(batch_size):
                for s in range(n_sources):
                    for c in range(n_channels):
                        self.randomize_parameters()
                        output[b, s, c, :] = self._apply_filter(x_np[b, s, c, :])

        # Restore original dtype
        if output.dtype != original_dtype:
            output = output.astype(original_dtype)

        # Convert back to torch if needed
        if was_torch:
            output = torch.from_numpy(output).to(original_device)
            # Handle NaN/inf
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

        return output


__all__ = [
    "PedalboardAdapter",
    "LowPassFilter",
    "HighPassFilter",
    "PeakFilter",
]
