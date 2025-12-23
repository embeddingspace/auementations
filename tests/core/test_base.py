"""BDD-style tests for base augmentation functionality.

These tests follow the Given-When-Then pattern to ensure the base
augmentation interface works as expected.
"""

import pytest
import torch

from tests.conftest import MockAugmentation


class TestBaseAugmentationInitialization:
    """Test scenarios for initializing base augmentations."""

    def test_given_valid_sample_rate_when_initialized_then_stores_sample_rate(self):
        """Given a valid sample rate, when augmentation is initialized, then it stores the sample rate."""
        # Given
        sample_rate = 16000

        # When
        aug = MockAugmentation(sample_rate=sample_rate)

        # Then
        assert aug.sample_rate == sample_rate

    def test_given_invalid_sample_rate_when_initialized_then_raises_error(self):
        """Given an invalid sample rate, when augmentation is initialized, then it raises ValueError."""
        # Given / When / Then
        with pytest.raises(ValueError, match="sample_rate must be a positive integer"):
            MockAugmentation(sample_rate=-1)

        with pytest.raises(ValueError, match="sample_rate must be a positive integer"):
            MockAugmentation(sample_rate=0)

    def test_given_probability_out_of_range_when_initialized_then_raises_error(self):
        """Given a probability outside [0, 1], when augmentation is initialized, then it raises ValueError."""
        # Given / When / Then
        with pytest.raises(ValueError, match="p must be between 0.0 and 1.0"):
            MockAugmentation(sample_rate=16000, p=1.5)

        with pytest.raises(ValueError, match="p must be between 0.0 and 1.0"):
            MockAugmentation(sample_rate=16000, p=-0.1)

    def test_given_valid_probability_when_initialized_then_stores_probability(self):
        """Given a valid probability, when augmentation is initialized, then it stores the probability."""
        # Given
        p = 0.7

        # When
        aug = MockAugmentation(sample_rate=16000, p=p)

        # Then
        assert aug.p == p

    def test_given_seed_when_initialized_then_enables_reproducibility(self):
        """Given a seed, when augmentation is initialized, then random operations are reproducible."""
        # Given
        seed = 42

        # When
        aug1 = MockAugmentation(sample_rate=16000, seed=seed)
        random1 = [aug1.should_apply() for _ in range(10)]

        aug2 = MockAugmentation(sample_rate=16000, seed=seed)
        random2 = [aug2.should_apply() for _ in range(10)]

        # Then
        assert random1 == random2


class TestBaseAugmentationProbabilisticBehavior:
    """Test scenarios for probabilistic application of augmentations."""

    def test_given_probability_one_when_applied_then_always_applies(self, mono_audio):
        """Given p=1.0, when augmentation is applied, then it always modifies audio."""
        # Given
        aug = MockAugmentation(sample_rate=16000, gain=2.0, p=1.0)
        original = mono_audio.clone()

        # When
        result = aug(mono_audio)

        # Then
        assert not torch.equal(result, original)
        assert torch.allclose(result, original * 2.0)

    def test_given_probability_zero_when_applied_then_never_applies(self, mono_audio):
        """Given p=0.0, when augmentation is applied, then it never modifies audio."""
        # Given
        aug = MockAugmentation(sample_rate=16000, gain=2.0, p=0.0)
        original = mono_audio.clone()

        # When
        result = aug(mono_audio)

        # Then
        assert torch.equal(result, original)

    def test_given_probability_half_when_applied_many_times_then_applies_approximately_half(
        self, mono_audio
    ):
        """Given p=0.5, when applied many times, then it applies approximately 50% of the time."""
        # Given
        aug = MockAugmentation(sample_rate=16000, gain=2.0, p=0.5, seed=42)
        n_trials = 1000

        # When
        apply_count = 0
        for _ in range(n_trials):
            if aug.should_apply():
                apply_count += 1

        # Then
        # Should be approximately 500, allow 10% deviation
        assert 450 <= apply_count <= 550


class TestBaseAugmentationConfiguration:
    """Test scenarios for augmentation configuration export."""

    def test_given_augmentation_when_exported_to_config_then_contains_all_parameters(
        self,
    ):
        """Given an augmentation, when exported to config, then config contains all parameters."""
        # Given
        aug = MockAugmentation(sample_rate=16000, gain=2.5, p=0.7, seed=42)

        # When
        config = aug.to_config()

        # Then
        assert "_target_" in config
        assert config["sample_rate"] == 16000
        assert config["p"] == 0.7
        assert config["seed"] == 42
        assert config["gain"] == 2.5

    def test_given_augmentation_when_repr_called_then_returns_readable_string(self):
        """Given an augmentation, when repr is called, then it returns a readable string."""
        # Given
        aug = MockAugmentation(sample_rate=16000, gain=2.0)

        # When
        repr_str = repr(aug)

        # Then
        assert "MockAugmentation" in repr_str
        assert "sample_rate=16000" in repr_str


class TestBaseAugmentationAudioProcessing:
    """Test scenarios for audio processing."""

    def test_given_mono_audio_when_processed_then_preserves_shape(self, mono_audio):
        """Given mono audio, when processed, then output shape matches input shape."""
        # Given
        aug = MockAugmentation(sample_rate=16000, gain=2.0)
        original_shape = mono_audio.shape

        # When
        result = aug(mono_audio)

        # Then
        assert result.shape == original_shape

    def test_given_stereo_audio_when_processed_then_preserves_shape(self, stereo_audio):
        """Given stereo audio, when processed, then output shape matches input shape."""
        # Given
        aug = MockAugmentation(sample_rate=16000, gain=2.0)
        original_shape = stereo_audio.shape

        # When
        result = aug(stereo_audio)

        # Then
        assert result.shape == original_shape

    def test_given_augmentation_when_called_multiple_times_then_counts_applications(
        self, mono_audio
    ):
        """Given an augmentation, when called multiple times, then it tracks application count."""
        # Given
        aug = MockAugmentation(sample_rate=16000, gain=2.0, p=1.0)

        # When
        for _ in range(5):
            aug(mono_audio)

        # Then
        assert aug.apply_count == 5
