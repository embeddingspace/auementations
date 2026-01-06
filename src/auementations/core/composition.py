"""Composition classes for building augmentation pipelines."""

from typing import Any, Dict, List, Optional, Union

from torch import Tensor

from auementations.config.config_store import auementations_store
from auementations.core.base import BaseAugmentation


class Composition(BaseAugmentation):
    """Base class for composition augmentations."""

    VALID_MODES = ["per_batch", "per_example"]

    def __init__(
        self,
        augmentations: dict[str, BaseAugmentation],
        sample_rate: int | float | None = None,
        p: float = 1.0,
        mode: str = "per_example",
        seed: Optional[int] = None,
    ):
        if not augmentations:
            raise ValueError("augmentations list cannot be empty")

        # Convert dict to list while preserving order and store names
        self.augmentations = augmentations

        aug_list = list(self.augmentations.values())

        # Infer sample_rate from first augmentation if not provided
        if sample_rate is None and aug_list[0].sample_rate is not None:
            sample_rate = aug_list[0].sample_rate

        super().__init__(sample_rate=sample_rate, p=p, seed=seed, mode=mode)

        # Validate that all augmentations have compatible sample rates
        augmentation_sample_rates = {
            aug.sample_rate for aug in aug_list if aug.sample_rate is not None
        }
        if len(augmentation_sample_rates) > 1:
            raise ValueError(
                f"All augmentations must have same sample_rate. "
                f"Expected {self.sample_rate}, got {augmentation_sample_rates}"
            )

        # Initialize effect with resolved parameters
        self.randomize_parameters()

    @property
    def names(self) -> list[str]:
        return list(self.augmentations.keys())

    def _init_composition(self):
        """Chooses how the composition should be applied for this round.

        By default, selects all of them.
        """
        return self.names

    def randomize_parameters(self) -> dict[str, Any]:
        """Sample the augmentations to apply, and get/set parameters for all the children."""
        self.selected_augmentations = self._init_composition()

        self.current_params = {}
        for name in self.selected_augmentations:
            self.current_params[name] = self.augmentations[name].randomize_parameters()

        return self.current_params

    def forward(self, audio: Tensor) -> Tensor:
        audio_ = audio.clone()
        for name in self.selected_augmentations:
            audio_ = self.augmentations[name].forward(audio_)

        return audio_

    def to_config(self) -> Dict[str, Any]:
        """Export configuration."""
        config = super().to_config()
        config["augmentations"] = [
            aug.to_config() for aug in self.augmentations.values()
        ]
        return config

    def __repr__(self) -> str:
        if self.names:
            # Show as dict with names
            items = [
                f'"{name}": {repr(aug)}' for name, aug in self.augmentations.items()
            ]
            aug_reprs = ",\n  ".join(items)
            return f"{self.__class__.__name__}({{\n  {aug_reprs}\n}})"
        else:
            # Show as list
            aug_reprs = ",\n  ".join(repr(aug) for aug in self.augmentations.values())
            return f"{self.__class__.__name__}([\n  {aug_reprs}\n])"


@auementations_store(name="compose", group="auementations/composition")
class Compose(Composition):
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
        super().__init__(
            augmentations, sample_rate=sample_rate, p=p, mode=mode, seed=seed
        )


@auementations_store(name="one_of", group="auementations/composition")
class OneOf(Composition):
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
        aug_list = list(augmentations.values())

        # Process weights
        if weights is None:
            self.weights = None
        else:
            # Convert dict weights to list
            if isinstance(weights, dict):
                if self.names is None:
                    raise ValueError("Cannot use dict weights with list augmentations")
                # Convert dict to list following augmentation order
                weight_list = [weights[name] for name in self.names]
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

        super().__init__(
            augmentations, sample_rate=sample_rate, p=p, mode=mode, seed=seed
        )

    def _init_composition(self):
        return self.rng.choice(self.names, 1, p=self.weights)

    def to_config(self) -> dict[str, Any]:
        config = super().to_config()
        config["weights"] = self.weights
        return config


@auementations_store(name="some_of", group="auementations/composition")
class SomeOf(Composition):
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
        self.replace = replace

        aug_list = list(augmentations.values())

        # Validate k
        self.k = k
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

        super().__init__(
            augmentations, sample_rate=sample_rate, p=p, mode=mode, seed=seed
        )

    def _init_composition(self):
        n_augmentations = self.rng.integers(self.k_min, self.k_max, endpoint=True)
        return self.rng.choice(self.names, n_augmentations, replace=self.replace)

    def to_config(self) -> dict[str, Any]:
        config = super().to_config()
        config["k"] = self.k
        config["replace"] = self.replace
        return config


__all__ = ["Compose", "OneOf", "SomeOf"]
