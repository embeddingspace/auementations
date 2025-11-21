"""BDD-style tests for parameter sampling functionality."""

import numpy as np
import pytest

from auementations.core.parameters import (
    ParameterSampler,
    ParameterValidator,
    resolve_parameter,
)


class TestParameterSamplerUniformDistribution:
    """Test scenarios for uniform parameter sampling."""

    def test_given_single_value_when_sampled_then_returns_same_value(self):
        """Given a single value, when sampled, then returns the same value."""
        # Given
        value = 0.5

        # When
        result = ParameterSampler.sample(value)

        # Then
        assert result == value

    def test_given_range_tuple_when_sampled_then_returns_value_in_range(self):
        """Given a range tuple, when sampled, then returns a value within the range."""
        # Given
        min_val, max_val = 0.0, 1.0

        # When
        result = ParameterSampler.sample((min_val, max_val))

        # Then
        assert min_val <= result <= max_val

    def test_given_range_tuple_when_sampled_many_times_then_uniformly_distributed(self):
        """Given a range tuple, when sampled many times, then values are uniformly distributed."""
        # Given
        np.random.seed(42)
        min_val, max_val = 0.0, 10.0
        n_samples = 10000

        # When
        samples = [ParameterSampler.sample((min_val, max_val)) for _ in range(n_samples)]

        # Then
        mean = np.mean(samples)
        assert 4.5 <= mean <= 5.5  # Should be close to 5.0 for uniform [0, 10]

    def test_given_invalid_range_when_sampled_then_raises_error(self):
        """Given an invalid range (min > max), when sampled, then raises ValueError."""
        # Given / When / Then
        with pytest.raises(ValueError, match="min .* must be <= max"):
            ParameterSampler.sample((10.0, 5.0))

    def test_given_integer_type_when_sampled_then_returns_integer(self):
        """Given dtype=int, when sampled, then returns integer value."""
        # Given
        value = 5.7

        # When
        result = ParameterSampler.sample(value, dtype=int)

        # Then
        assert isinstance(result, int)
        assert result == 5


class TestParameterSamplerLogUniformDistribution:
    """Test scenarios for log-uniform parameter sampling."""

    def test_given_log_uniform_spec_when_sampled_then_returns_value_in_range(self):
        """Given a log_uniform spec, when sampled, then returns value in specified range."""
        # Given
        spec = {"dist": "log_uniform", "min": 0.01, "max": 1.0}

        # When
        result = ParameterSampler.sample(spec)

        # Then
        assert 0.01 <= result <= 1.0

    def test_given_log_uniform_spec_when_sampled_many_times_then_log_distributed(self):
        """Given a log_uniform spec, when sampled many times, then logarithms are uniformly distributed."""
        # Given
        np.random.seed(42)
        spec = {"dist": "log_uniform", "min": 0.01, "max": 1.0}
        n_samples = 10000

        # When
        samples = [ParameterSampler.sample(spec) for _ in range(n_samples)]
        log_samples = np.log(samples)

        # Then
        # Log of samples should be uniform between log(0.01) and log(1.0)
        expected_log_mean = (np.log(0.01) + np.log(1.0)) / 2
        log_mean = np.mean(log_samples)
        assert abs(log_mean - expected_log_mean) < 0.2

    def test_given_log_uniform_with_negative_values_when_sampled_then_raises_error(self):
        """Given log_uniform with negative values, when sampled, then raises ValueError."""
        # Given
        spec = {"dist": "log_uniform", "min": -1.0, "max": 1.0}

        # When / Then
        with pytest.raises(ValueError, match="log_uniform requires positive"):
            ParameterSampler.sample(spec)


class TestParameterSamplerNormalDistribution:
    """Test scenarios for normal distribution sampling."""

    def test_given_normal_spec_when_sampled_then_returns_value(self):
        """Given a normal distribution spec, when sampled, then returns a value."""
        # Given
        spec = {"dist": "normal", "mean": 0.0, "std": 1.0}

        # When
        result = ParameterSampler.sample(spec)

        # Then
        assert isinstance(result, float)

    def test_given_normal_spec_when_sampled_many_times_then_follows_distribution(self):
        """Given a normal spec, when sampled many times, then follows normal distribution."""
        # Given
        np.random.seed(42)
        spec = {"dist": "normal", "mean": 5.0, "std": 2.0}
        n_samples = 10000

        # When
        samples = [ParameterSampler.sample(spec) for _ in range(n_samples)]

        # Then
        mean = np.mean(samples)
        std = np.std(samples)
        assert 4.8 <= mean <= 5.2  # Close to 5.0
        assert 1.8 <= std <= 2.2    # Close to 2.0


class TestParameterSamplerTruncatedNormalDistribution:
    """Test scenarios for truncated normal distribution sampling."""

    def test_given_truncated_normal_when_sampled_then_returns_value_in_range(self):
        """Given a truncated_normal spec, when sampled, then returns value within bounds."""
        # Given
        spec = {"dist": "truncated_normal", "mean": 0.0, "std": 1.0, "min": -2.0, "max": 2.0}

        # When
        result = ParameterSampler.sample(spec)

        # Then
        assert -2.0 <= result <= 2.0

    def test_given_truncated_normal_when_sampled_many_times_then_all_in_range(self):
        """Given a truncated_normal spec, when sampled many times, then all values are in range."""
        # Given
        np.random.seed(42)
        spec = {"dist": "truncated_normal", "mean": 0.0, "std": 1.0, "min": -1.5, "max": 1.5}
        n_samples = 1000

        # When
        samples = [ParameterSampler.sample(spec) for _ in range(n_samples)]

        # Then
        assert all(-1.5 <= s <= 1.5 for s in samples)


class TestParameterSamplerEdgeCases:
    """Test scenarios for edge cases and error handling."""

    def test_given_unsupported_distribution_when_sampled_then_raises_error(self):
        """Given an unsupported distribution, when sampled, then raises ValueError."""
        # Given
        spec = {"dist": "exponential", "lambda": 1.0}

        # When / Then
        with pytest.raises(ValueError, match="Unsupported distribution"):
            ParameterSampler.sample(spec)

    def test_given_unsupported_type_when_sampled_then_raises_error(self):
        """Given an unsupported parameter type, when sampled, then raises TypeError."""
        # Given
        value = [0.1, 0.2, 0.3]  # List is not supported

        # When / Then
        with pytest.raises(TypeError, match="Unsupported parameter type"):
            ParameterSampler.sample(value)


class TestParameterValidator:
    """Test scenarios for parameter validation."""

    def test_given_value_in_range_when_validated_then_passes(self):
        """Given a value within range, when validated, then passes without error."""
        # Given
        value = 5.0

        # When / Then (should not raise)
        ParameterValidator.validate_range(value, min_val=0.0, max_val=10.0)

    def test_given_value_below_minimum_when_validated_then_raises_error(self):
        """Given a value below minimum, when validated, then raises ValueError."""
        # Given
        value = -1.0

        # When / Then
        with pytest.raises(ValueError, match="must be >= 0.0"):
            ParameterValidator.validate_range(value, min_val=0.0, name="test_param")

    def test_given_value_above_maximum_when_validated_then_raises_error(self):
        """Given a value above maximum, when validated, then raises ValueError."""
        # Given
        value = 11.0

        # When / Then
        with pytest.raises(ValueError, match="must be <= 10.0"):
            ParameterValidator.validate_range(value, max_val=10.0, name="test_param")

    def test_given_valid_probability_when_validated_then_passes(self):
        """Given a valid probability [0, 1], when validated, then passes."""
        # Given / When / Then
        ParameterValidator.validate_probability(0.0)
        ParameterValidator.validate_probability(0.5)
        ParameterValidator.validate_probability(1.0)

    def test_given_invalid_probability_when_validated_then_raises_error(self):
        """Given an invalid probability, when validated, then raises ValueError."""
        # Given / When / Then
        with pytest.raises(ValueError):
            ParameterValidator.validate_probability(-0.1)

        with pytest.raises(ValueError):
            ParameterValidator.validate_probability(1.1)

    def test_given_valid_sample_rate_when_validated_then_passes(self):
        """Given a valid sample rate, when validated, then passes."""
        # Given / When / Then
        ParameterValidator.validate_sample_rate(16000)
        ParameterValidator.validate_sample_rate(44100)

    def test_given_invalid_sample_rate_when_validated_then_raises_error(self):
        """Given an invalid sample rate, when validated, then raises error."""
        # Given / When / Then
        with pytest.raises(ValueError, match="must be positive"):
            ParameterValidator.validate_sample_rate(0)

        with pytest.raises(ValueError, match="must be positive"):
            ParameterValidator.validate_sample_rate(-16000)

        with pytest.raises(TypeError, match="must be int"):
            ParameterValidator.validate_sample_rate(16000.5)


class TestResolveParameterConvenience:
    """Test scenarios for the convenience resolve_parameter function."""

    def test_given_any_parameter_spec_when_resolved_then_returns_value(self):
        """Given any parameter spec, when resolved, then returns a value."""
        # Given / When / Then
        assert resolve_parameter(5.0) == 5.0
        assert 0.0 <= resolve_parameter((0.0, 1.0)) <= 1.0
        assert 0.01 <= resolve_parameter({"dist": "log_uniform", "min": 0.01, "max": 1.0}) <= 1.0
