"""Parameter sampling utilities for probabilistic augmentations."""

from typing import Any, Dict, Tuple, Union

import numpy as np

# Type alias for parameter values that can be single values or ranges
ParameterValue = Union[float, int, Tuple[float, float], Tuple[int, int], Dict[str, Any]]


class ParameterSampler:
    """Handles sampling of parameters from various distributions.

    Supports:
    - Single values (no sampling)
    - Uniform ranges: (min, max)
    - Distribution specs: {"dist": "log_uniform", "min": 0.01, "max": 1.0}
    """

    @staticmethod
    def sample(
        value: ParameterValue,
        rng: np.random.Generator,
        dtype: type = float,
    ) -> float | int:
        """Sample a parameter value.

        Args:
            value: Parameter specification. Can be:
                - Single value: returned as-is
                - Tuple (min, max): uniformly sampled
                - Dict with 'dist' key: sampled from specified distribution
            rng: NumPy random generator for reproducible sampling.
            dtype: Target data type (float or int)

        Returns:
            Sampled parameter value.

        Examples:
            >>> rng = np.random.default_rng(42)
            >>> ParameterSampler.sample(0.5, rng)
            0.5
            >>> ParameterSampler.sample((0.0, 1.0), rng)  # Random value in [0, 1]
            0.7234...
            >>> ParameterSampler.sample({"dist": "log_uniform", "min": 0.01, "max": 1.0}, rng)
            0.0543...
        """
        # Single value - no sampling
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return dtype(value)

        # Tuple range - uniform sampling
        if isinstance(value, tuple) and len(value) == 2:
            min_val, max_val = value
            if min_val > max_val:
                raise ValueError(f"min ({min_val}) must be <= max ({max_val})")

            sampled = rng.uniform(min_val, max_val)
            return dtype(sampled)

        # Dictionary distribution spec
        if isinstance(value, dict):
            return ParameterSampler._sample_from_distribution(value, rng, dtype)

        raise TypeError(f"Unsupported parameter type: {type(value)}")

    @staticmethod
    def _sample_from_distribution(
        spec: Dict[str, Any],
        rng: np.random.Generator,
        dtype: type,
    ) -> float | int:
        """Sample from a distribution specified by a dictionary.

        Args:
            spec: Distribution specification with 'dist' key and parameters.
            rng: NumPy random generator for reproducible sampling.
            dtype: Target data type.

        Returns:
            Sampled value.

        Supported distributions:
            - uniform: {"dist": "uniform", "min": 0, "max": 1}
            - log_uniform: {"dist": "log_uniform", "min": 0.01, "max": 1}
            - normal: {"dist": "normal", "mean": 0, "std": 1}
            - truncated_normal: {"dist": "truncated_normal", "mean": 0, "std": 1, "min": -2, "max": 2}
        """
        dist = spec.get("dist")

        if dist == "uniform":
            min_val = spec["min"]
            max_val = spec["max"]
            sampled = rng.uniform(min_val, max_val)
            return dtype(sampled)

        elif dist == "log_uniform":
            min_val = spec["min"]
            max_val = spec["max"]
            if min_val <= 0 or max_val <= 0:
                raise ValueError("log_uniform requires positive min and max values")
            log_min = np.log(min_val)
            log_max = np.log(max_val)
            sampled = np.exp(rng.uniform(log_min, log_max))
            return dtype(sampled)

        elif dist == "normal":
            mean = spec["mean"]
            std = spec["std"]
            sampled = rng.normal(mean, std)
            return dtype(sampled)

        elif dist == "truncated_normal":
            mean = spec["mean"]
            std = spec["std"]
            min_val = spec.get("min", -np.inf)
            max_val = spec.get("max", np.inf)

            # Sample until we get a value in range (rejection sampling)
            # Add safety limit to avoid infinite loops
            for _ in range(1000):
                sampled = rng.normal(mean, std)
                if min_val <= sampled <= max_val:
                    return dtype(sampled)

            # Fallback to clamping if rejection sampling fails
            sampled = np.clip(rng.normal(mean, std), min_val, max_val)
            return dtype(sampled)

        else:
            raise ValueError(f"Unsupported distribution: {dist}")


class ParameterValidator:
    """Validates parameter specifications and values."""

    @staticmethod
    def validate_range(
        value: float | int,
        min_val: float | int | None = None,
        max_val: float | int | None = None,
        name: str = "parameter",
    ) -> None:
        """Validate that a value is within specified range.

        Args:
            value: Value to validate.
            min_val: Minimum allowed value (inclusive). None means no minimum.
            max_val: Maximum allowed value (inclusive). None means no maximum.
            name: Parameter name for error messages.

        Raises:
            ValueError: If value is outside the specified range.
        """
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")

    @staticmethod
    def validate_probability(p: float, name: str = "p") -> None:
        """Validate that a probability value is in [0, 1].

        Args:
            p: Probability value to validate.
            name: Parameter name for error messages.

        Raises:
            ValueError: If p is not in [0, 1].
        """
        ParameterValidator.validate_range(p, 0.0, 1.0, name)

    @staticmethod
    def validate_sample_rate(sample_rate: int) -> None:
        """Validate sample rate.

        Args:
            sample_rate: Sample rate in Hz.

        Raises:
            ValueError: If sample rate is invalid.
        """
        if not isinstance(sample_rate, int):
            raise TypeError(f"sample_rate must be int, got {type(sample_rate)}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")


def resolve_parameter(
    value: ParameterValue,
    rng: np.random.Generator,
    dtype: type = float,
) -> Union[float, int]:
    """Convenience function to resolve a parameter value.

    This is a shorthand for ParameterSampler.sample().

    Args:
        value: Parameter specification.
        rng: NumPy random generator for reproducible sampling.
        dtype: Target data type.

    Returns:
        Resolved parameter value.
    """
    return ParameterSampler.sample(value, rng, dtype)
