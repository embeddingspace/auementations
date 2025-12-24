"""Base abstractions for audio augmentations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from torch import Tensor


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

    VALID_MODES = ["per_batch", "per_example", "per_source", "per_channel"]

    def __init__(
        self,
        sample_rate: int | float,
        p: float = 1.0,
        mode: str = "per_example",
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

        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.mode = mode

        self.sample_rate = sample_rate
        self.p = p
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def __call__(
        self, audio: Tensor, log: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Optional[Dict]]:
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
        log_obj = None
        output = audio

        if self.mode == "per_example" and len(audio.shape) == 4:
            log_obj = []

            output = audio.clone()
            for i in range(audio.shape[0]):
                item_log = None
                if self.should_apply():
                    params = self.randomize_parameters()
                    item_log = self._create_log_dict(params)

                    output[i] = self.forward(output[i])
                log_obj.append(item_log)

        elif self.mode == "per_source":
            raise NotImplementedError()
        elif self.mode == "per_channel":
            raise NotImplementedError()

        else:  # elif self.mode == "per_batch":
            log_obj = {}

            if self.should_apply():
                params = self.randomize_parameters()
                log_obj = self._create_log_dict(params)

                # Call forward
                output = self.forward(audio)

        if not log:
            return output
        return output, log_obj

    @abstractmethod
    def forward(self, audio: Tensor) -> Tensor:
        """Applies the augmentation to the audio or subset of audio selected by the mode."""
        pass

    def randomize_parameters(self) -> None:
        """Sample random parameters from configured distributions.

        This method is called automatically before each augmentation application
        when parameters are specified as ranges. Override this method to implement
        custom parameter sampling logic.
        """
        return {}

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
