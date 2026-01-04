"""BDD-style tests for pedalboard adapter NaN and inf handling.

These tests ensure that PedalboardAdapter properly sanitizes invalid
values (NaN, inf, -inf) that may be produced by pedalboard effects.
"""

from functools import partial
import numpy as np
import torch


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
        audio = torch.randn(1, 1, 16000).to(torch.float32)

        # Create adapter with mock effect that returns NaN
        aug = PedalboardAdapter(
            effect_class=lambda: MockPedalboardEffect(
                partial(np.full_like, fill_value=np.nan)
            ),
            sample_rate=sample_rate,
            p=1.0,
        )

        # When
        result = aug(audio)

        # Then - all NaN values should be replaced with 0.0
        assert not torch.isnan(result).any(), "Output should not contain NaN values"
        assert torch.allclose(result, torch.tensor(0.0)), (
            "NaN values should be replaced with 0.0"
        )

    def test_given_effect_produces_positive_inf_when_applied_then_returns_valid_values(
        self,
    ):
        """Given a pedalboard effect that produces positive inf, when applied, then inf values are replaced with 1.0."""
        # Given
        from auementations.adapters.pedalboard import PedalboardAdapter

        sample_rate = 16000
        audio = torch.randn(1, 16000).to(torch.float32)

        aug = PedalboardAdapter(
            effect_class=lambda: MockPedalboardEffect(
                partial(np.full_like, fill_value=np.inf)
            ),
            sample_rate=sample_rate,
            p=1.0,
        )

        # When
        result = aug(audio)

        # Then - all positive inf values should be replaced with 1.0
        assert not torch.isinf(result).any(), "Output should not contain inf values"
        assert torch.allclose(result, torch.tensor(1.0)), (
            "Positive inf values should be replaced with 1.0"
        )

    def test_given_effect_produces_negative_inf_when_applied_then_returns_valid_values(
        self,
    ):
        """Given a pedalboard effect that produces negative inf, when applied, then inf values are replaced with -1.0."""
        # Given
        from auementations.adapters.pedalboard import PedalboardAdapter

        sample_rate = 16000
        audio = torch.randn(1, 16000).to(torch.float32)

        aug = PedalboardAdapter(
            effect_class=lambda: MockPedalboardEffect(
                partial(np.full_like, fill_value=-np.inf)
            ),
            sample_rate=sample_rate,
            p=1.0,
        )

        # When
        result = aug(audio)

        # Then - all negative inf values should be replaced with -1.0
        assert not torch.isinf(result).any(), "Output should not contain inf values"
        assert torch.allclose(result, torch.tensor(-1.0)), (
            "Negative inf values should be replaced with -1.0"
        )

    def test_given_effect_produces_mixed_invalid_values_when_applied_then_all_sanitized(
        self,
    ):
        """Given a pedalboard effect that produces mixed NaN/inf values, when applied, then all are sanitized."""
        # Given
        from auementations.adapters.pedalboard import PedalboardAdapter

        sample_rate = 16000
        audio = torch.randn(2, sample_rate)

        # Create output with mixed invalid values
        def fn(audio):
            invalid_output = np.zeros_like(audio)
            invalid_output[:4000] = np.nan  # First quarter: NaN
            invalid_output[4000:8000] = np.inf  # Second quarter: +inf
            invalid_output[8000:12000] = -np.inf  # Third quarter: -inf
            invalid_output[12000:] = 0.5  # Last quarter: valid
            return invalid_output

        aug = PedalboardAdapter(
            effect_class=lambda: MockPedalboardEffect(fn),
            sample_rate=sample_rate,
            p=1.0,
        )

        # When
        result = aug(audio)

        # Then - no invalid values in output
        assert not torch.isnan(result).any(), "Output should not contain NaN"
        assert not torch.isinf(result).any(), "Output should not contain inf"
        # Check expected replacements in first channel
        assert torch.allclose(result[0, :4000], torch.tensor(0.0)), (
            "NaN should become 0.0"
        )
        assert torch.allclose(result[0, 4000:8000], torch.tensor(1.0)), (
            "+inf should become 1.0"
        )
        assert torch.allclose(result[0, 8000:12000], torch.tensor(-1.0)), (
            "-inf should become -1.0"
        )
        assert torch.allclose(result[0, 12000:], torch.tensor(0.5)), (
            "Valid values should be preserved"
        )

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
        audio = torch.randn(batch_size, 1, 8000).float()

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

        # When
        result = aug(audio)

        # Then
        assert not torch.isnan(result).any(), "Output should not contain NaN"
        assert not torch.isinf(result).any(), "Output should not contain inf"
        assert torch.allclose(result[0], torch.tensor(0.0)), (
            "First batch NaN should become 0.0"
        )
        assert torch.allclose(result[1], torch.tensor(1.0)), (
            "Second batch +inf should become 1.0"
        )
        assert torch.allclose(result[2], torch.tensor(-1.0)), (
            "Third batch -inf should become -1.0"
        )
