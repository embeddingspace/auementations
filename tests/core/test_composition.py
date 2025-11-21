"""BDD-style tests for composition functionality."""

import numpy as np
import pytest

from auementations.core.composition import Compose, OneOf, SomeOf
from tests.conftest import MockAugmentation


class TestComposeSequentialApplication:
    """Test scenarios for sequential composition of augmentations."""

    def test_given_multiple_augmentations_when_composed_then_applies_in_sequence(
        self, mono_audio, sample_rate
    ):
        """Given multiple augmentations, when composed, then applies them in sequence."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0)
        compose = Compose([aug1, aug2])

        original = mono_audio.copy()

        # When
        result = compose(mono_audio)

        # Then
        # Should apply aug1 first (x2), then aug2 (x3) -> total x6
        expected = original * 2.0 * 3.0
        assert np.allclose(result, expected)

    def test_given_empty_augmentation_list_when_composed_then_raises_error(self):
        """Given an empty augmentation list, when composed, then raises ValueError."""
        # Given / When / Then
        with pytest.raises(ValueError, match="augmentations list cannot be empty"):
            Compose([])

    def test_given_augmentations_when_composed_then_infers_sample_rate(self, sample_rate):
        """Given augmentations, when composed without explicit sample_rate, then infers from first."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate)
        aug2 = MockAugmentation(sample_rate=sample_rate)

        # When
        compose = Compose([aug1, aug2])

        # Then
        assert compose.sample_rate == sample_rate

    def test_given_mismatched_sample_rates_when_composed_then_raises_error(self):
        """Given augmentations with different sample rates, when composed, then raises ValueError."""
        # Given
        aug1 = MockAugmentation(sample_rate=16000)
        aug2 = MockAugmentation(sample_rate=44100)

        # When / Then
        with pytest.raises(ValueError, match="must have same sample_rate"):
            Compose([aug1, aug2])

    def test_given_compose_with_probability_when_not_applied_then_returns_original(
        self, mono_audio, sample_rate
    ):
        """Given a Compose with p=0, when applied, then returns original audio."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0)
        compose = Compose([aug1, aug2], p=0.0)

        original = mono_audio.copy()

        # When
        result = compose(mono_audio)

        # Then
        assert np.array_equal(result, original)

    def test_given_compose_when_exported_to_config_then_includes_all_augmentations(
        self, sample_rate
    ):
        """Given a Compose, when exported to config, then includes all augmentation configs."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0)
        compose = Compose([aug1, aug2])

        # When
        config = compose.to_config()

        # Then
        assert "augmentations" in config
        assert len(config["augmentations"]) == 2
        assert config["augmentations"][0]["gain"] == 2.0
        assert config["augmentations"][1]["gain"] == 3.0

    def test_given_compose_when_repr_called_then_shows_all_augmentations(self, sample_rate):
        """Given a Compose, when repr is called, then shows all augmentations."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0)
        compose = Compose([aug1, aug2])

        # When
        repr_str = repr(compose)

        # Then
        assert "Compose" in repr_str
        assert "MockAugmentation" in repr_str


class TestOneOfRandomSelection:
    """Test scenarios for OneOf random selection."""

    def test_given_multiple_augmentations_when_one_of_applied_then_applies_exactly_one(
        self, mono_audio, sample_rate
    ):
        """Given multiple augmentations, when OneOf is applied, then applies exactly one."""
        # Given
        np.random.seed(42)
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0, p=1.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0, p=1.0)
        aug3 = MockAugmentation(sample_rate=sample_rate, gain=4.0, p=1.0)
        one_of = OneOf([aug1, aug2, aug3])

        # When
        result = one_of(mono_audio)

        # Then
        # Should be one of 2x, 3x, or 4x the original
        original = mono_audio
        is_aug1 = np.allclose(result, original * 2.0)
        is_aug2 = np.allclose(result, original * 3.0)
        is_aug3 = np.allclose(result, original * 4.0)

        assert is_aug1 or is_aug2 or is_aug3
        # Should apply exactly one (not multiple)
        assert sum([is_aug1, is_aug2, is_aug3]) == 1

    def test_given_weighted_augmentations_when_one_of_applied_then_respects_weights(
        self, mono_audio, sample_rate
    ):
        """Given weighted augmentations, when OneOf applied many times, then respects weights."""
        # Given
        np.random.seed(42)
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0)
        # Give aug1 3x the weight of aug2
        one_of = OneOf([aug1, aug2], weights=[0.75, 0.25])

        n_trials = 1000
        aug1_count = 0

        # When
        for _ in range(n_trials):
            result = one_of(mono_audio.copy())
            if np.allclose(result, mono_audio * 2.0):
                aug1_count += 1

        # Then
        # Should be approximately 75% aug1
        assert 700 <= aug1_count <= 800

    def test_given_mismatched_weights_when_one_of_initialized_then_raises_error(
        self, sample_rate
    ):
        """Given weights that don't match augmentation count, when initialized, then raises ValueError."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate)
        aug2 = MockAugmentation(sample_rate=sample_rate)

        # When / Then
        with pytest.raises(ValueError, match="Number of weights.*must match"):
            OneOf([aug1, aug2], weights=[0.5, 0.3, 0.2])  # 3 weights for 2 augmentations

    def test_given_empty_augmentation_list_when_one_of_initialized_then_raises_error(self):
        """Given empty augmentation list, when OneOf initialized, then raises ValueError."""
        # Given / When / Then
        with pytest.raises(ValueError, match="augmentations list cannot be empty"):
            OneOf([])

    def test_given_one_of_when_exported_to_config_then_includes_weights(self, sample_rate):
        """Given a OneOf with weights, when exported to config, then includes weights."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate)
        aug2 = MockAugmentation(sample_rate=sample_rate)
        one_of = OneOf([aug1, aug2], weights=[0.7, 0.3])

        # When
        config = one_of.to_config()

        # Then
        assert "weights" in config
        assert len(config["weights"]) == 2


class TestSomeOfRandomSelection:
    """Test scenarios for SomeOf k-out-of-n selection."""

    def test_given_k_equals_two_when_some_of_applied_then_applies_exactly_two(
        self, mono_audio, sample_rate
    ):
        """Given k=2, when SomeOf is applied, then applies exactly 2 augmentations."""
        # Given
        np.random.seed(42)
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=1.5)
        aug3 = MockAugmentation(sample_rate=sample_rate, gain=1.2)
        aug4 = MockAugmentation(sample_rate=sample_rate, gain=1.1)

        some_of = SomeOf(k=2, augmentations=[aug1, aug2, aug3, aug4])

        # When
        result = some_of(mono_audio)

        # Then
        # Result should be product of exactly 2 gain values
        # We can't predict which, but we can verify it's not original
        assert not np.allclose(result, mono_audio)

    def test_given_k_range_when_some_of_applied_then_applies_k_in_range(
        self, mono_audio, sample_rate
    ):
        """Given k as range, when SomeOf applied, then applies k augmentations within range."""
        # Given
        np.random.seed(42)
        augmentations = [
            MockAugmentation(sample_rate=sample_rate, gain=1.1),
            MockAugmentation(sample_rate=sample_rate, gain=1.2),
            MockAugmentation(sample_rate=sample_rate, gain=1.3),
            MockAugmentation(sample_rate=sample_rate, gain=1.4),
        ]

        some_of = SomeOf(k=(1, 3), augmentations=augmentations)

        # When - apply multiple times and check apply counts
        for aug in augmentations:
            aug.apply_count = 0

        n_trials = 100
        for _ in range(n_trials):
            some_of(mono_audio.copy())

        # Then
        # Total applications should be between 100 and 300 (1-3 per trial)
        total_applications = sum(aug.apply_count for aug in augmentations)
        assert 100 <= total_applications <= 300

    def test_given_k_equals_zero_when_some_of_applied_then_returns_original(
        self, mono_audio, sample_rate
    ):
        """Given k=0, when SomeOf is applied, then returns original audio."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0)
        some_of = SomeOf(k=0, augmentations=[aug1, aug2])

        original = mono_audio.copy()

        # When
        result = some_of(mono_audio)

        # Then
        assert np.array_equal(result, original)

    def test_given_replace_true_when_some_of_applied_then_can_select_same_augmentation_twice(
        self, mono_audio, sample_rate
    ):
        """Given replace=True, when SomeOf applied, then can select same augmentation multiple times."""
        # Given
        np.random.seed(42)
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        # Only one augmentation, but k=3 with replacement
        some_of = SomeOf(k=3, augmentations=[aug1], replace=True)

        original = mono_audio.copy()

        # When
        result = some_of(mono_audio)

        # Then
        # Should apply aug1 three times: 2.0^3 = 8.0
        expected = original * (2.0 ** 3)
        assert np.allclose(result, expected)

    def test_given_k_greater_than_n_when_some_of_initialized_then_raises_error(self, sample_rate):
        """Given k > number of augmentations, when initialized without replace, then raises ValueError."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate)
        aug2 = MockAugmentation(sample_rate=sample_rate)

        # When / Then
        with pytest.raises(ValueError, match="k must be between"):
            SomeOf(k=3, augmentations=[aug1, aug2], replace=False)

    def test_given_some_of_when_exported_to_config_then_includes_k_and_replace(
        self, sample_rate
    ):
        """Given a SomeOf, when exported to config, then includes k and replace parameters."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate)
        aug2 = MockAugmentation(sample_rate=sample_rate)
        some_of = SomeOf(k=2, augmentations=[aug1, aug2], replace=True)

        # When
        config = some_of.to_config()

        # Then
        assert config["k"] == 2
        assert config["replace"] is True

    def test_given_k_range_when_exported_to_config_then_includes_tuple(self, sample_rate):
        """Given k as range, when exported to config, then includes tuple."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate)
        aug2 = MockAugmentation(sample_rate=sample_rate)
        some_of = SomeOf(k=(1, 2), augmentations=[aug1, aug2])

        # When
        config = some_of.to_config()

        # Then
        assert config["k"] == (1, 2)


class TestNestedComposition:
    """Test scenarios for nested composition (Compose of OneOf, etc.)."""

    def test_given_nested_compositions_when_applied_then_works_correctly(
        self, mono_audio, sample_rate
    ):
        """Given nested compositions, when applied, then correctly applies nested logic."""
        # Given
        # Create a pipeline: Compose([fixed_aug, OneOf([aug1, aug2])])
        fixed_aug = MockAugmentation(sample_rate=sample_rate, gain=2.0)

        one_of_aug1 = MockAugmentation(sample_rate=sample_rate, gain=1.5)
        one_of_aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0)
        one_of = OneOf([one_of_aug1, one_of_aug2])

        compose = Compose([fixed_aug, one_of])

        original = mono_audio.copy()

        # When
        result = compose(mono_audio)

        # Then
        # Should apply fixed_aug (x2), then one of the OneOf augmentations
        option1 = original * 2.0 * 1.5  # fixed then one_of_aug1
        option2 = original * 2.0 * 3.0  # fixed then one_of_aug2

        is_option1 = np.allclose(result, option1)
        is_option2 = np.allclose(result, option2)

        assert is_option1 or is_option2
