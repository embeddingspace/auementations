"""Composition classes for building augmentation pipelines."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from auementations.config.config_store import auementations_store
from auementations.core.base import BaseAugmentation


@auementations_store(name="compose", group="composition")
class Compose(BaseAugmentation):
    """Sequential composition of augmentations.

    Applies multiple augmentations in sequence, passing the output of each
    augmentation as input to the next.

    Example:
        >>> compose = Compose([
        ...     Gain(sample_rate=16000, min_gain_db=-6, max_gain_db=6),
        ...     PitchShift(sample_rate=16000, min_semitones=-2, max_semitones=2),
        ... ])
        >>> augmented = compose(audio)
    """

    def __init__(
        self,
        augmentations: List[BaseAugmentation],
        sample_rate: Optional[int] = None,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize sequential composition.

        Args:
            augmentations: List of augmentations to apply in sequence.
            sample_rate: Sample rate. If None, inferred from first augmentation.
            p: Probability of applying the entire composition.
            seed: Random seed for reproducibility.
        """
        if not augmentations:
            raise ValueError("augmentations list cannot be empty")

        # Infer sample_rate from first augmentation if not provided
        if sample_rate is None:
            sample_rate = augmentations[0].sample_rate

        super().__init__(sample_rate=sample_rate, p=p, seed=seed)
        self.augmentations = augmentations

        # Validate that all augmentations have compatible sample rates
        for aug in augmentations:
            if aug.sample_rate != self.sample_rate:
                raise ValueError(
                    f"All augmentations must have same sample_rate. "
                    f"Expected {self.sample_rate}, got {aug.sample_rate} "
                    f"for {aug.__class__.__name__}"
                )

    def __call__(
        self, audio: Union[np.ndarray, Any], **kwargs
    ) -> Union[np.ndarray, Any]:
        """Apply augmentations sequentially.

        Args:
            audio: Input audio.
            **kwargs: Additional parameters passed to each augmentation.

        Returns:
            Augmented audio.
        """
        if not self.should_apply():
            return audio

        result = audio
        for augmentation in self.augmentations:
            augmentation.randomize_parameters()
            result = augmentation(result, **kwargs)

        return result

    def to_config(self) -> Dict[str, Any]:
        """Export configuration."""
        config = super().to_config()
        config["augmentations"] = [aug.to_config() for aug in self.augmentations]
        return config

    def __repr__(self) -> str:
        aug_reprs = ",\n  ".join(repr(aug) for aug in self.augmentations)
        return f"Compose([\n  {aug_reprs}\n])"


@auementations_store(name="one_of", group="composition")
class OneOf(BaseAugmentation):
    """Randomly select and apply one augmentation from a list.

    Useful for applying mutually exclusive augmentations, like different
    types of noise or different time-stretching factors.

    Example:
        >>> one_of = OneOf([
        ...     TimeStretch(sample_rate=16000, min_rate=0.8, max_rate=0.9),
        ...     TimeStretch(sample_rate=16000, min_rate=1.1, max_rate=1.2),
        ...     Identity(sample_rate=16000),
        ... ])
        >>> augmented = one_of(audio)  # Applies exactly one
    """

    def __init__(
        self,
        augmentations: List[BaseAugmentation],
        weights: Optional[List[float]] = None,
        sample_rate: Optional[int] = None,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize OneOf composition.

        Args:
            augmentations: List of augmentations to choose from.
            weights: Optional weights for each augmentation. If None, uniform.
            sample_rate: Sample rate. If None, inferred from first augmentation.
            p: Probability of applying any augmentation (if False, returns input unchanged).
            seed: Random seed for reproducibility.
        """
        if not augmentations:
            raise ValueError("augmentations list cannot be empty")

        if sample_rate is None:
            sample_rate = augmentations[0].sample_rate

        super().__init__(sample_rate=sample_rate, p=p, seed=seed)
        self.augmentations = augmentations

        # Validate sample rates
        for aug in augmentations:
            if aug.sample_rate != self.sample_rate:
                raise ValueError(
                    f"All augmentations must have same sample_rate. "
                    f"Expected {self.sample_rate}, got {aug.sample_rate}"
                )

        # Process weights
        if weights is None:
            self.weights = None
        else:
            if len(weights) != len(augmentations):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of augmentations ({len(augmentations)})"
                )
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def __call__(
        self, audio: Union[np.ndarray, Any], **kwargs
    ) -> Union[np.ndarray, Any]:
        """Apply one randomly selected augmentation.

        Args:
            audio: Input audio.
            **kwargs: Additional parameters passed to the selected augmentation.

        Returns:
            Augmented audio.
        """
        if not self.should_apply():
            return audio

        # Select one augmentation
        if self.weights is None:
            idx = np.random.randint(0, len(self.augmentations))
        else:
            idx = np.random.choice(len(self.augmentations), p=self.weights)

        selected = self.augmentations[idx]
        selected.randomize_parameters()
        return selected(audio, **kwargs)

    def to_config(self) -> Dict[str, Any]:
        """Export configuration."""
        config = super().to_config()
        config["augmentations"] = [aug.to_config() for aug in self.augmentations]
        if self.weights is not None:
            config["weights"] = self.weights
        return config


@auementations_store(name="some_of", group="composition")
class SomeOf(BaseAugmentation):
    """Apply k randomly selected augmentations from a list.

    This allows applying multiple random augmentations without applying all of them.

    Example:
        >>> some_of = SomeOf(
        ...     k=2,
        ...     augmentations=[
        ...         Gain(sample_rate=16000, min_gain_db=-6, max_gain_db=6),
        ...         HighPassFilter(sample_rate=16000, min_cutoff_freq=20, max_cutoff_freq=400),
        ...         LowPassFilter(sample_rate=16000, min_cutoff_freq=4000, max_cutoff_freq=8000),
        ...         AddNoise(sample_rate=16000, min_snr_db=10, max_snr_db=30),
        ...     ]
        ... )
        >>> augmented = some_of(audio)  # Applies exactly 2 random augmentations
    """

    def __init__(
        self,
        k: int,
        augmentations: List[BaseAugmentation],
        replace: bool = False,
        sample_rate: Optional[int] = None,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize SomeOf composition.

        Args:
            k: Number of augmentations to apply. Can be:
                - Integer: exact number
                - Tuple (min, max): random number in range
            augmentations: List of augmentations to choose from.
            replace: If True, same augmentation can be selected multiple times.
            sample_rate: Sample rate. If None, inferred from first augmentation.
            p: Probability of applying the composition.
            seed: Random seed for reproducibility.
        """
        if not augmentations:
            raise ValueError("augmentations list cannot be empty")

        if sample_rate is None:
            sample_rate = augmentations[0].sample_rate

        super().__init__(sample_rate=sample_rate, p=p, seed=seed)
        self.augmentations = augmentations
        self.replace = replace

        # Validate k
        if isinstance(k, int):
            if k < 0:
                raise ValueError(f"k must be non-negative, got {k}")
            # Only validate upper bound if not using replacement
            if not replace and k > len(augmentations):
                raise ValueError(
                    f"k must be between 0 and {len(augmentations)} when replace=False, got {k}"
                )
            self.k_min = self.k_max = k
        elif isinstance(k, tuple) and len(k) == 2:
            self.k_min, self.k_max = k
            if self.k_min < 0:
                raise ValueError(f"k_min must be non-negative, got {self.k_min}")
            # Only validate upper bound if not using replacement
            if not replace and self.k_max > len(augmentations):
                raise ValueError(
                    f"k range must be between 0 and {len(augmentations)} when replace=False, got {k}"
                )
        else:
            raise TypeError(f"k must be int or tuple of two ints, got {type(k)}")

        # Validate sample rates
        for aug in augmentations:
            if aug.sample_rate != self.sample_rate:
                raise ValueError(
                    f"All augmentations must have same sample_rate. "
                    f"Expected {self.sample_rate}, got {aug.sample_rate}"
                )

    def __call__(
        self, audio: Union[np.ndarray, Any], **kwargs
    ) -> Union[np.ndarray, Any]:
        """Apply k randomly selected augmentations.

        Args:
            audio: Input audio.
            **kwargs: Additional parameters passed to each augmentation.

        Returns:
            Augmented audio.
        """
        if not self.should_apply():
            return audio

        # Determine how many augmentations to apply
        if self.k_min == self.k_max:
            k = self.k_min
        else:
            k = np.random.randint(self.k_min, self.k_max + 1)

        if k == 0:
            return audio

        # Select k augmentations
        if self.replace:
            indices = np.random.choice(len(self.augmentations), size=k, replace=True)
        else:
            indices = np.random.choice(len(self.augmentations), size=k, replace=False)

        # Apply selected augmentations sequentially
        result = audio
        for idx in indices:
            selected = self.augmentations[idx]
            selected.randomize_parameters()
            result = selected(result, **kwargs)

        return result

    def to_config(self) -> Dict[str, Any]:
        """Export configuration."""
        config = super().to_config()
        config["k"] = (
            self.k_min if self.k_min == self.k_max else (self.k_min, self.k_max)
        )
        config["augmentations"] = [aug.to_config() for aug in self.augmentations]
        config["replace"] = self.replace
        return config


__all__ = ["Compose", "OneOf", "SomeOf"]
