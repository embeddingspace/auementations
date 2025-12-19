"""BDD-style tests for pedalboard adapter NaN and inf handling.

These tests ensure that PedalboardAdapter properly sanitizes invalid
values (NaN, inf, -inf) that may be produced by pedalboard effects.
"""

import numpy as np


class MockPedalboardEffect:
    """Mock pedalboard effect that returns invalid values for testing."""

    def __init__(self, return_value):
        self.return_value = return_value

    def process(self, audio, sample_rate):
        """Return the configured invalid value."""
        if callable(self.return_value):
            return self.return_value(audio)
        return self.return_value


class TestPedalboardAdapterNaNAndInfHandling:
    """Test scenarios for handling NaN and inf values from pedalboard effects."""

    def test_given_effect_produces_nan_when_applied_then_returns_valid_values(self):
        """Given a pedalboard effect that produces NaN, when applied, then NaN values are replaced with 0.0."""
        # Given
        from auementations.adapters.pedalboard import PedalboardAdapter

        sample_rate = 16000
        audio = np.random.randn(1, 16000).astype(np.float32)

        # Create adapter with mock effect that returns NaN
        aug = PedalboardAdapter(
            effect_class=lambda: MockPedalboardEffect(np.full_like(audio, np.nan)),
            sample_rate=sample_rate,
            p=1.0,
        )
        # Override the effect to ensure it returns NaN
        aug.effect = MockPedalboardEffect(np.full_like(audio, np.nan))

        # When
        result = aug(audio)

        # Then - all NaN values should be replaced with 0.0
        assert not np.isnan(result).any(), "Output should not contain NaN values"
        assert np.allclose(result, 0.0), "NaN values should be replaced with 0.0"

    def test_given_effect_produces_positive_inf_when_applied_then_returns_valid_values(
        self,
    ):
        """Given a pedalboard effect that produces positive inf, when applied, then inf values are replaced with 1.0."""
        # Given
        from auementations.adapters.pedalboard import PedalboardAdapter

        sample_rate = 16000
        audio = np.random.randn(1, 16000).astype(np.float32)

        aug = PedalboardAdapter(
            effect_class=lambda: MockPedalboardEffect(np.full_like(audio, np.inf)),
            sample_rate=sample_rate,
            p=1.0,
        )
        aug.effect = MockPedalboardEffect(np.full_like(audio, np.inf))

        # When
        result = aug(audio)

        # Then - all positive inf values should be replaced with 1.0
        assert not np.isinf(result).any(), "Output should not contain inf values"
        assert np.allclose(result, 1.0), (
            "Positive inf values should be replaced with 1.0"
        )

    def test_given_effect_produces_negative_inf_when_applied_then_returns_valid_values(
        self,
    ):
        """Given a pedalboard effect that produces negative inf, when applied, then inf values are replaced with -1.0."""
        # Given
        from auementations.adapters.pedalboard import PedalboardAdapter

        sample_rate = 16000
        audio = np.random.randn(1, 16000).astype(np.float32)

        aug = PedalboardAdapter(
            effect_class=lambda: MockPedalboardEffect(np.full_like(audio, -np.inf)),
            sample_rate=sample_rate,
            p=1.0,
        )
        aug.effect = MockPedalboardEffect(np.full_like(audio, -np.inf))

        # When
        result = aug(audio)

        # Then - all negative inf values should be replaced with -1.0
        assert not np.isinf(result).any(), "Output should not contain inf values"
        assert np.allclose(result, -1.0), (
            "Negative inf values should be replaced with -1.0"
        )

    def test_given_effect_produces_mixed_invalid_values_when_applied_then_all_sanitized(
        self,
    ):
        """Given a pedalboard effect that produces mixed NaN/inf values, when applied, then all are sanitized."""
        # Given
        from auementations.adapters.pedalboard import PedalboardAdapter

        sample_rate = 16000
        audio = np.random.randn(2, 16000).astype(np.float32)

        # Create output with mixed invalid values
        invalid_output = np.zeros_like(audio)
        invalid_output[0, :4000] = np.nan  # First quarter: NaN
        invalid_output[0, 4000:8000] = np.inf  # Second quarter: +inf
        invalid_output[0, 8000:12000] = -np.inf  # Third quarter: -inf
        invalid_output[0, 12000:] = 0.5  # Last quarter: valid
        invalid_output[1, :] = audio[1, :]  # Second channel: copy input

        aug = PedalboardAdapter(
            effect_class=lambda: MockPedalboardEffect(invalid_output.copy()),
            sample_rate=sample_rate,
            p=1.0,
        )
        aug.effect = MockPedalboardEffect(invalid_output.copy())

        # When
        result = aug(audio)

        # Then - no invalid values in output
        assert not np.isnan(result).any(), "Output should not contain NaN"
        assert not np.isinf(result).any(), "Output should not contain inf"
        # Check expected replacements in first channel
        assert np.allclose(result[0, :4000], 0.0), "NaN should become 0.0"
        assert np.allclose(result[0, 4000:8000], 1.0), "+inf should become 1.0"
        assert np.allclose(result[0, 8000:12000], -1.0), "-inf should become -1.0"
        assert np.allclose(result[0, 12000:], 0.5), "Valid values should be preserved"

    def test_given_torch_tensor_with_invalid_values_when_applied_then_returns_valid_tensor(
        self,
    ):
        """Given torch tensor input and effect produces invalid values, when applied, then returns valid torch tensor."""
        pytest = __import__("pytest")
        torch = pytest.importorskip("torch")

        # Given
        from auementations.adapters.pedalboard import PedalboardAdapter

        sample_rate = 16000
        audio = torch.randn(1, 16000, dtype=torch.float32)

        # Create invalid output (as numpy since that's what pedalboard returns)
        invalid_output = np.full((1, 16000), np.nan, dtype=np.float32)

        aug = PedalboardAdapter(
            effect_class=lambda: MockPedalboardEffect(invalid_output.copy()),
            sample_rate=sample_rate,
            p=1.0,
        )
        aug.effect = MockPedalboardEffect(invalid_output.copy())

        # When
        result = aug(audio)

        # Then
        assert isinstance(result, torch.Tensor), "Should return torch tensor"
        assert not torch.isnan(result).any(), "Output should not contain NaN"
        assert not torch.isinf(result).any(), "Output should not contain inf"
        assert torch.allclose(result, torch.zeros_like(result)), (
            "NaN should be replaced with 0.0"
        )

    def test_given_batch_with_invalid_values_when_applied_then_all_sanitized(self):
        """Given batched input and effect produces invalid values, when applied, then all batch items are sanitized."""
        # Given
        from auementations.adapters.pedalboard import PedalboardAdapter

        sample_rate = 16000
        batch_size = 3
        audio = np.random.randn(batch_size, 1, 8000).astype(np.float32)

        # Create a function that returns different invalid values for each call
        call_count = [0]

        def get_invalid_output(audio_in):
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                return np.full_like(audio_in, np.nan)
            elif idx == 1:
                return np.full_like(audio_in, np.inf)
            else:
                return np.full_like(audio_in, -np.inf)

        aug = PedalboardAdapter(
            effect_class=lambda: MockPedalboardEffect(get_invalid_output),
            sample_rate=sample_rate,
            p=1.0,
            seed=42,
        )
        aug.effect = MockPedalboardEffect(get_invalid_output)

        # When
        result = aug(audio)

        # Then
        assert not np.isnan(result).any(), "Output should not contain NaN"
        assert not np.isinf(result).any(), "Output should not contain inf"
        assert np.allclose(result[0], 0.0), "First batch NaN should become 0.0"
        assert np.allclose(result[1], 1.0), "Second batch +inf should become 1.0"
        assert np.allclose(result[2], -1.0), "Third batch -inf should become -1.0"
