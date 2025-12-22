"""Composition classes for building augmentation pipelines."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from auementations.config.config_store import auementations_store
from auementations.core.base import BaseAugmentation


@auementations_store(name="compose", group="auementations/composition")
class Compose(BaseAugmentation):
    """Sequential composition of augmentations.

    Applies multiple augmentations in sequence, passing the output of each
    augmentation as input to the next.

    Example with dict (better for structured configs):
        >>> compose = Compose({
        ...     "gain": Gain(sample_rate=16000, min_gain_db=-6, max_gain_db=6),
        ...     "pitch": PitchShift(sample_rate=16000, min_semitones=-2, max_semitones=2),
        ... })
        >>> augmented = compose(audio)
    """

    VALID_MODES = ["per_batch", "per_example"]

    def __init__(
        self,
        augmentations: dict[str, BaseAugmentation],
        sample_rate: int | float | None = None,
        p: float = 1.0,
        mode: str = "per_example",
        seed: Optional[int] = None,
    ):
        """Initialize sequential composition.

        Args:
            augmentations: List or dict of augmentations to apply in sequence.
                          If dict, preserves insertion order (like OrderedDict in Python 3.7+).
            sample_rate: Sample rate. If None, inferred from first augmentation.
            p: Probability of applying the entire composition.
            mode: Composition mode - "per_example" (independent randomization per batch example) or "per_batch" (same randomization for all).
            seed: Random seed for reproducibility.
        """
        if not augmentations:
            raise ValueError("augmentations list cannot be empty")

        # Validate mode
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        # Convert dict to list while preserving order and store names
        self.augmentation_names = list(augmentations.keys())
        aug_list = list(augmentations.values())

        # Infer sample_rate from first augmentation if not provided
        if sample_rate is None and aug_list[0].sample_rate is not None:
            sample_rate = aug_list[0].sample_rate

        super().__init__(sample_rate=sample_rate, p=p, seed=seed)
        self.augmentations = aug_list
        self.mode = mode

        # Validate that all augmentations have compatible sample rates
        for i, aug in enumerate(aug_list):
            if aug.sample_rate != self.sample_rate:
                name = self.augmentation_names[i] if self.augmentation_names else str(i)
                raise ValueError(
                    f"All augmentations must have same sample_rate. "
                    f"Expected {self.sample_rate}, got {aug.sample_rate} "
                    f"for augmentation '{name}' ({aug.__class__.__name__})"
                )

    def __call__(self, audio: np.ndarray | Any, **kwargs) -> np.ndarray | Any:
        """Apply augmentations sequentially.

        Args:
            audio: Input audio.
            **kwargs: Additional parameters passed to each augmentation.

        Returns:
            Augmented audio.
        """
        if not self.should_apply():
            return audio

        # Check if we need per_example mode
        if (
            self.mode == "per_example"
            and isinstance(audio, np.ndarray)
            and audio.ndim >= 2
        ):
            # Apply augmentations with independent randomization for each example
            batch_size = audio.shape[0]
            results = []

            for i in range(batch_size):
                # Apply the full sequence to each example independently
                result = audio[i]
                for augmentation in self.augmentations:
                    augmentation.randomize_parameters()
                    result = augmentation(result, **kwargs)
                results.append(result)

            # Stack results back into batch
            return np.stack(results, axis=0)
        else:
            # per_batch mode: apply same randomization to entire batch
            result = audio
            for augmentation in self.augmentations:
                augmentation.randomize_parameters()
                result = augmentation(result, **kwargs)

            return result

    def to_config(self) -> Dict[str, Any]:
        """Export configuration."""
        config = super().to_config()
        config["augmentations"] = [aug.to_config() for aug in self.augmentations]
        config["mode"] = self.mode
        return config

    def __repr__(self) -> str:
        if self.augmentation_names:
            # Show as dict with names
            items = [
                f'"{name}": {repr(aug)}'
                for name, aug in zip(self.augmentation_names, self.augmentations)
            ]
            aug_reprs = ",\n  ".join(items)
            return f"Compose({{\n  {aug_reprs}\n}})"
        else:
            # Show as list
            aug_reprs = ",\n  ".join(repr(aug) for aug in self.augmentations)
            return f"Compose([\n  {aug_reprs}\n])"


@auementations_store(name="one_of", group="auementations/composition")
class OneOf(BaseAugmentation):
    """Randomly select and apply one augmentation from a list or dict.

    Useful for applying mutually exclusive augmentations, like different
    types of noise or different time-stretching factors.

    Example with dict (better for structured configs):
        >>> one_of = OneOf({
        ...     "attenuate": Gain(sample_rate=16000, min_gain_db=-12, max_gain_db=0),
        ...     "amplify": Gain(sample_rate=16000, min_gain_db=0, max_gain_db=12),
        ... })
        >>> augmented = one_of(audio)  # Applies exactly one
    """

    VALID_MODES = ["per_batch", "per_example"]

    def __init__(
        self,
        augmentations: dict[str, BaseAugmentation],
        weights: Optional[Union[List[float], Dict[str, float]]] = None,
        sample_rate: int | float | None = None,
        p: float = 1.0,
        mode: str = "per_example",
        seed: Optional[int] = None,
    ):
        """Initialize OneOf composition.

        Args:
            augmentations: List or dict of augmentations to choose from.
                          If dict, preserves insertion order.
            weights: Optional weights for each augmentation. If None, uniform.
                    Can be a list (matching augmentation order) or dict (matching augmentation names).
            sample_rate: Sample rate. If None, inferred from first augmentation.
            p: Probability of applying any augmentation (if False, returns input unchanged).
            mode: Composition mode - "per_example" (different aug per batch example) or "per_batch" (same aug for all).
            seed: Random seed for reproducibility.
        """
        if not augmentations:
            raise ValueError("augmentations list cannot be empty")

        # Validate mode
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        # Convert dict to list while preserving order and store names
        self.augmentation_names = list(augmentations.keys())
        aug_list = list(augmentations.values())

        if sample_rate is None and aug_list[0].sample_rate is not None:
            sample_rate = aug_list[0].sample_rate

        super().__init__(sample_rate=sample_rate, p=p, seed=seed)
        self.augmentations = aug_list
        self.mode = mode

        # Validate sample rates
        for i, aug in enumerate(aug_list):
            if aug.sample_rate != self.sample_rate:
                name = self.augmentation_names[i] if self.augmentation_names else str(i)
                raise ValueError(
                    f"All augmentations must have same sample_rate. "
                    f"Expected {self.sample_rate}, got {aug.sample_rate} "
                    f"for augmentation '{name}'"
                )

        # Process weights
        if weights is None:
            self.weights = None
        else:
            # Convert dict weights to list
            if isinstance(weights, dict):
                if self.augmentation_names is None:
                    raise ValueError("Cannot use dict weights with list augmentations")
                # Convert dict to list following augmentation order
                weight_list = [weights[name] for name in self.augmentation_names]
            else:
                weight_list = weights

            if len(weight_list) != len(aug_list):
                raise ValueError(
                    f"Number of weights ({len(weight_list)}) must match "
                    f"number of augmentations ({len(aug_list)})"
                )
            # Normalize weights
            total = sum(weight_list)
            self.weights = [w / total for w in weight_list]

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

        # Check if we need per_example mode
        if (
            self.mode == "per_example"
            and isinstance(audio, np.ndarray)
            and audio.ndim >= 2
        ):
            # Apply different augmentation to each example in the batch
            batch_size = audio.shape[0]
            results = []

            for i in range(batch_size):
                # Select one augmentation for this example
                if self.weights is None:
                    idx = np.random.randint(0, len(self.augmentations))
                else:
                    idx = np.random.choice(len(self.augmentations), p=self.weights)

                selected = self.augmentations[idx]
                selected.randomize_parameters()
                # Apply to single example
                example = audio[i]
                result = selected(example, **kwargs)
                results.append(result)

            # Stack results back into batch
            return np.stack(results, axis=0)
        else:
            # per_batch mode: apply same augmentation to entire batch
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
        config["mode"] = self.mode
        if self.weights is not None:
            config["weights"] = self.weights
        return config


@auementations_store(name="some_of", group="auementations/composition")
class SomeOf(BaseAugmentation):
    """Apply k randomly selected augmentations from a list or dict.

    This allows applying multiple random augmentations without applying all of them.

    Example with dict (better for structured configs):
        >>> some_of = SomeOf(
        ...     k=2,
        ...     augmentations={
        ...         "gain": Gain(sample_rate=16000, min_gain_db=-6, max_gain_db=6),
        ...         "hpf": HighPassFilter(sample_rate=16000, min_cutoff_freq=20, max_cutoff_freq=400),
        ...         "lpf": LowPassFilter(sample_rate=16000, min_cutoff_freq=4000, max_cutoff_freq=8000),
        ...     }
        ... )
        >>> augmented = some_of(audio)  # Applies exactly 2 random augmentations
    """

    VALID_MODES = ["per_batch", "per_example"]

    def __init__(
        self,
        k: Union[int, tuple],
        augmentations: dict[str, BaseAugmentation],
        replace: bool = False,
        sample_rate: int | float | None = None,
        p: float = 1.0,
        mode: str = "per_example",
        seed: Optional[int] = None,
    ):
        """Initialize SomeOf composition.

        Args:
            k: Number of augmentations to apply. Can be:
                - Integer: exact number
                - Tuple (min, max): random number in range
            augmentations: List or dict of augmentations to choose from.
                          If dict, preserves insertion order.
            replace: If True, same augmentation can be selected multiple times.
            sample_rate: Sample rate. If None, inferred from first augmentation.
            p: Probability of applying the composition.
            mode: Composition mode - "per_example" (different selection per batch example) or "per_batch" (same selection for all).
            seed: Random seed for reproducibility.
        """
        if not augmentations:
            raise ValueError("augmentations list cannot be empty")

        # Validate mode
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        # Convert dict to list while preserving order and store names
        self.augmentation_names = list(augmentations.keys())
        aug_list = list(augmentations.values())

        if sample_rate is None and aug_list[0].sample_rate is not None:
            sample_rate = aug_list[0].sample_rate

        super().__init__(sample_rate=sample_rate, p=p, seed=seed)
        self.augmentations = aug_list
        self.replace = replace
        self.mode = mode

        # Validate k
        if isinstance(k, int):
            if k < 0:
                raise ValueError(f"k must be non-negative, got {k}")
            # Only validate upper bound if not using replacement
            if not replace and k > len(aug_list):
                raise ValueError(
                    f"k must be between 0 and {len(aug_list)} when replace=False, got {k}"
                )
            self.k_min = self.k_max = k
        elif isinstance(k, tuple) and len(k) == 2:
            self.k_min, self.k_max = k
            if self.k_min < 0:
                raise ValueError(f"k_min must be non-negative, got {self.k_min}")
            # Only validate upper bound if not using replacement
            if not replace and self.k_max > len(aug_list):
                raise ValueError(
                    f"k range must be between 0 and {len(aug_list)} when replace=False, got {k}"
                )
        else:
            raise TypeError(f"k must be int or tuple of two ints, got {type(k)}")

        # Validate sample rates
        for i, aug in enumerate(aug_list):
            if aug.sample_rate != self.sample_rate:
                name = self.augmentation_names[i] if self.augmentation_names else str(i)
                raise ValueError(
                    f"All augmentations must have same sample_rate. "
                    f"Expected {self.sample_rate}, got {aug.sample_rate} "
                    f"for augmentation '{name}'"
                )

    def __call__(self, audio: np.ndarray | Any, **kwargs) -> np.ndarray | Any:
        """Apply k randomly selected augmentations.

        Args:
            audio: Input audio.
            **kwargs: Additional parameters passed to each augmentation.

        Returns:
            Augmented audio.
        """
        if not self.should_apply():
            return audio

        # Check if we need per_example mode
        if (
            self.mode == "per_example"
            and isinstance(audio, np.ndarray)
            and audio.ndim >= 2
        ):
            # Apply different k augmentations to each example in the batch
            batch_size = audio.shape[0]
            results = []

            for i in range(batch_size):
                # Determine how many augmentations to apply for this example
                if self.k_min == self.k_max:
                    k = self.k_min
                else:
                    k = np.random.randint(self.k_min, self.k_max + 1)

                if k == 0:
                    results.append(audio[i])
                    continue

                # Select k augmentations for this example
                if self.replace:
                    indices = np.random.choice(
                        len(self.augmentations), size=k, replace=True
                    )
                else:
                    indices = np.random.choice(
                        len(self.augmentations), size=k, replace=False
                    )

                # Apply selected augmentations sequentially to this example
                result = audio[i]
                for idx in indices:
                    selected = self.augmentations[idx]
                    selected.randomize_parameters()
                    result = selected(result, **kwargs)

                results.append(result)

            # Stack results back into batch
            return np.stack(results, axis=0)
        else:
            # per_batch mode: apply same k augmentations to entire batch
            # Determine how many augmentations to apply
            if self.k_min == self.k_max:
                k = self.k_min
            else:
                k = np.random.randint(self.k_min, self.k_max + 1)

            if k == 0:
                return audio

            # Select k augmentations
            if self.replace:
                indices = np.random.choice(
                    len(self.augmentations), size=k, replace=True
                )
            else:
                indices = np.random.choice(
                    len(self.augmentations), size=k, replace=False
                )

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
        config["mode"] = self.mode
        return config


__all__ = ["Compose", "OneOf", "SomeOf"]
