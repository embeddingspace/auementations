"""Base abstractions for audio augmentations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np


class BaseAugmentation(ABC):
    """Base class for all audio augmentations.

    All augmentations must implement this interface to ensure consistency
    across different backends (torch_audiomentations, pedalboard, etc.).

    Key design principles:
    - sample_rate is always explicit, never inferred
    - Parameters can be ranges (tuples) for probabilistic sampling
    - All augmentations support probability of application (p)
    - Reproducible randomness through seed support
    """

    def __init__(
        self,
        sample_rate: int | float,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize base augmentation.

        Args:
            sample_rate: Audio sample rate in Hz. Must be explicitly provided.
            p: Probability of applying this augmentation (0.0 to 1.0).
            seed: Random seed for reproducibility. If None, uses global random state.
        """
        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be a positive integer, got {sample_rate}"
            )

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be between 0.0 and 1.0, got {p}")

        self.sample_rate = sample_rate
        self.p = p
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def __call__(
        self, audio: Union[np.ndarray, Any], log: bool = False, **kwargs
    ) -> Union[np.ndarray, Any, tuple[Union[np.ndarray, Any], Optional[Dict]]]:
        """Apply augmentation to audio.

        Args:
            audio: Audio data as numpy array or backend-specific tensor.
                   Shape: (num_channels, num_samples) or (num_samples,)
            log: If True, return (audio, log_dict) tuple. If False, return audio only.
            **kwargs: Additional backend-specific parameters.

        Returns:
            If log=False: Augmented audio in same format as input.
            If log=True: Tuple of (augmented_audio, log_dict) where log_dict contains
                        information about the augmentation that was applied.
                        log_dict is None if augmentation was not applied (p check failed).
        """
        pass

    def randomize_parameters(self) -> None:
        """Sample random parameters from configured distributions.

        This method is called automatically before each augmentation application
        when parameters are specified as ranges. Override this method to implement
        custom parameter sampling logic.
        """
        pass

    def should_apply(self) -> bool:
        """Determine whether to apply this augmentation based on probability.

        Returns:
            True if augmentation should be applied, False otherwise.
        """
        return np.random.random() < self.p

    def _create_log_dict(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a log dictionary for this augmentation.

        Args:
            parameters: Dictionary of parameter names and values that were used.

        Returns:
            Dictionary containing augmentation name and parameters.
        """
        return {
            "augmentation": self.__class__.__name__,
            "parameters": parameters,
        }

    def to_config(self) -> Dict[str, Any]:
        """Export augmentation configuration as dictionary.

        Useful for saving/loading augmentation pipelines and Hydra integration.

        Returns:
            Dictionary containing all configuration parameters.
        """
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "sample_rate": self.sample_rate,
            "p": self.p,
            "seed": self.seed,
        }

    def __repr__(self) -> str:
        """String representation of augmentation."""
        params = ", ".join(
            f"{k}={v}" for k, v in self.to_config().items() if k != "_target_"
        )
        return f"{self.__class__.__name__}({params})"
