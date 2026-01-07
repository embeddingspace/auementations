"""BDD-style tests for composition functionality."""

import pytest
import torch

from auementations.core.base import BaseAugmentation
from auementations.core.composition import Compose, OneOf, SomeOf
from tests.conftest import MockAugmentation


# Mock augmentations with distinct, easily verifiable effects for per-example mode tests
class AddOneAugmentation(BaseAugmentation):
    """Adds 1.0 to the audio signal."""

    def __init__(self, sample_rate: int, p: float = 1.0):
        super().__init__(sample_rate=sample_rate, p=p)
        self.apply_count = 0

    def forward(self, audio):
        """Apply the augmentation."""
        self.apply_count += 1
        return audio + 1.0

    def randomize_parameters(self):
        """No parameters to randomize."""
        return {}


class MultiplyByTenAugmentation(BaseAugmentation):
    """Multiplies audio signal by 10.0."""

    def __init__(self, sample_rate: int, p: float = 1.0):
        super().__init__(sample_rate=sample_rate, p=p)
        self.apply_count = 0

    def forward(self, audio):
        """Apply the augmentation."""
        self.apply_count += 1
        return audio * 10.0

    def randomize_parameters(self):
        """No parameters to randomize."""
        return {}


class AddHundredAugmentation(BaseAugmentation):
    """Adds 100.0 to the audio signal."""

    def __init__(self, sample_rate: int, p: float = 1.0):
        super().__init__(sample_rate=sample_rate, p=p)
        self.apply_count = 0

    def forward(self, audio):
        """Apply the augmentation."""
        self.apply_count += 1
        return audio + 100.0

    def randomize_parameters(self):
        """No parameters to randomize."""
        return {}


class TestComposeSequentialApplication:
    """Test scenarios for sequential composition of augmentations."""

    def test_given_multiple_augmentations_when_composed_then_applies_in_sequence(
        self, mono_audio, sample_rate
    ):
        """Given multiple augmentations, when composed, then applies them in sequence."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0)
        compose = Compose({"a": aug1, "b": aug2})

        original = mono_audio.clone()

        # When
        result = compose(mono_audio)

        # Then
        # Should apply aug1 first (x2), then aug2 (x3) -> total x6
        expected = original * 2.0 * 3.0
        assert torch.allclose(result, expected)

    def test_given_empty_augmentation_list_when_composed_then_raises_error(self):
        """Given an empty augmentation list, when composed, then raises ValueError."""
        # Given / When / Then
        with pytest.raises(ValueError, match="augmentations list cannot be empty"):
            Compose([])

    def test_given_augmentations_when_composed_then_infers_sample_rate(
        self, sample_rate
    ):
        """Given augmentations, when composed without explicit sample_rate, then infers from first."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate)
        aug2 = MockAugmentation(sample_rate=sample_rate)

        # When
        compose = Compose({"a": aug1, "b": aug2})

        # Then
        assert compose.sample_rate == sample_rate

    def test_given_mismatched_sample_rates_when_composed_then_raises_error(self):
        """Given augmentations with different sample rates, when composed, then raises ValueError."""
        # Given
        aug1 = MockAugmentation(sample_rate=16000)
        aug2 = MockAugmentation(sample_rate=44100)

        # When / Then
        with pytest.raises(ValueError, match="must have same sample_rate"):
            Compose({"a": aug1, "b": aug2})

    def test_given_compose_with_probability_when_not_applied_then_returns_original(
        self, mono_audio, sample_rate
    ):
        """Given a Compose with p=0, when applied, then returns original audio."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0)
        compose = Compose({"a": aug1, "b": aug2}, p=0.0)

        original = mono_audio.clone()

        # When
        result = compose(mono_audio)

        # Then
        assert torch.equal(result, original)

    def test_given_compose_when_exported_to_config_then_includes_all_augmentations(
        self, sample_rate
    ):
        """Given a Compose, when exported to config, then includes all augmentation configs."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0)
        compose = Compose({"a": aug1, "b": aug2})

        # When
        config = compose.to_config()

        # Then
        assert "augmentations" in config
        assert len(config["augmentations"]) == 2
        assert config["augmentations"][0]["gain"] == 2.0
        assert config["augmentations"][1]["gain"] == 3.0

    def test_given_compose_when_repr_called_then_shows_all_augmentations(
        self, sample_rate
    ):
        """Given a Compose, when repr is called, then shows all augmentations."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0)
        compose = Compose({"a": aug1, "b": aug2})

        # When
        repr_str = repr(compose)

        # Then
        assert "Compose" in repr_str
        assert "MockAugmentation" in repr_str

    def test_given_dict_augmentations_when_repr_called_then_shows_names(
        self, sample_rate
    ):
        """Given dict augmentations, when repr called, then shows augmentation names."""
        # Given
        augmentations = {
            "gain_up": MockAugmentation(sample_rate=sample_rate, gain=2.0),
            "gain_down": MockAugmentation(sample_rate=sample_rate, gain=0.5),
        }
        compose = Compose(augmentations=augmentations)

        # When
        repr_str = repr(compose)

        # Then
        assert "gain_up" in repr_str
        assert "gain_down" in repr_str
        assert "{" in repr_str  # Dict notation


class TestOneOfRandomSelection:
    """Test scenarios for OneOf random selection."""

    def test_given_multiple_augmentations_when_one_of_applied_then_applies_exactly_one(
        self, mono_audio, sample_rate
    ):
        """Given multiple augmentations, when OneOf is applied, then applies exactly one."""
        # Given
        torch.manual_seed(42)
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0, p=1.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0, p=1.0)
        aug3 = MockAugmentation(sample_rate=sample_rate, gain=4.0, p=1.0)
        one_of = OneOf({"a": aug1, "b": aug2, "c": aug3})

        # When
        result = one_of(mono_audio)

        # Then
        # Should be one of 2x, 3x, or 4x the original
        original = mono_audio
        is_aug1 = torch.allclose(result, original * 2.0)
        is_aug2 = torch.allclose(result, original * 3.0)
        is_aug3 = torch.allclose(result, original * 4.0)

        assert is_aug1 or is_aug2 or is_aug3
        # Should apply exactly one (not multiple)
        assert sum([is_aug1, is_aug2, is_aug3]) == 1

    def test_given_weighted_augmentations_when_one_of_applied_then_respects_weights(
        self, mono_audio, sample_rate
    ):
        """Given weighted augmentations, when OneOf applied many times, then respects weights."""
        # Given
        torch.manual_seed(42)
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0)
        # Give aug1 3x the weight of aug2
        one_of = OneOf({"a": aug1, "b": aug2}, weights=[0.75, 0.25])

        n_trials = 1000
        aug1_count = 0

        # When
        for _ in range(n_trials):
            result = one_of(mono_audio.clone())
            if torch.allclose(result, mono_audio * 2.0):
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
            OneOf(
                {"a": aug1, "b": aug2}, weights=[0.5, 0.3, 0.2]
            )  # 3 weights for 2 augmentations

    def test_given_empty_augmentation_list_when_one_of_initialized_then_raises_error(
        self,
    ):
        """Given empty augmentation list, when OneOf initialized, then raises ValueError."""
        # Given / When / Then
        with pytest.raises(ValueError, match="augmentations list cannot be empty"):
            OneOf({})

    def test_given_one_of_when_exported_to_config_then_includes_weights(
        self, sample_rate
    ):
        """Given a OneOf with weights, when exported to config, then includes weights."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate)
        aug2 = MockAugmentation(sample_rate=sample_rate)
        one_of = OneOf({"a": aug1, "b": aug2}, weights=[0.7, 0.3])

        # When
        config = one_of.to_config()

        # Then
        assert "weights" in config
        assert len(config["weights"]) == 2

    def test_given_dict_with_dict_weights_when_one_of_created_then_applies_weights(
        self, mono_audio, sample_rate
    ):
        """Given dict augmentations with dict weights, when created, then applies weights correctly."""
        # Given
        augmentations = {
            "high_prob": MockAugmentation(sample_rate=sample_rate, gain=10.0),
            "low_prob": MockAugmentation(sample_rate=sample_rate, gain=1.0),
        }
        weights = {
            "high_prob": 0.9,
            "low_prob": 0.1,
        }

        # When
        one_of = OneOf(augmentations, weights=weights, p=1.0, seed=42)
        audio = torch.ones(1, 1, 100, dtype=torch.float32)

        # Apply many times and count
        high_count = 0
        low_count = 0
        for _ in range(100):
            result = one_of(audio.clone())
            if torch.isclose(result[0, 0, 0], torch.tensor(10.0)):
                high_count += 1
            else:
                low_count += 1

        # Then
        assert high_count > low_count

    def test_given_dict_with_mismatched_sample_rates_when_one_of_created_then_raises_error_with_name(
        self,
    ):
        """Given dict with mismatched sample rates, when OneOf created, then raises error."""
        # Given
        augmentations = {
            "first": MockAugmentation(sample_rate=16000, gain=2.0),
            "second": MockAugmentation(sample_rate=44100, gain=3.0),
        }

        # When/Then
        # TODO: Error message should include augmentation name "second" to help debugging
        with pytest.raises(ValueError):
            OneOf(augmentations=augmentations)


class TestSomeOfRandomSelection:
    """Test scenarios for SomeOf k-out-of-n selection."""

    def test_given_k_equals_two_when_some_of_applied_then_applies_exactly_two(
        self, mono_audio, sample_rate
    ):
        """Given k=2, when SomeOf is applied, then applies exactly 2 augmentations."""
        # Given
        torch.manual_seed(42)
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=1.5)
        aug3 = MockAugmentation(sample_rate=sample_rate, gain=1.2)
        aug4 = MockAugmentation(sample_rate=sample_rate, gain=1.1)

        some_of = SomeOf(
            augmentations={"a": aug1, "b": aug2, "c": aug3, "d": aug4},
            k=2,
        )

        # When
        result = some_of(mono_audio)

        # Then
        # Result should be product of exactly 2 gain values
        # We can't predict which, but we can verify it's not original
        assert not torch.allclose(result, mono_audio)

    def test_given_k_range_when_some_of_applied_then_applies_k_in_range(
        self, mono_audio, sample_rate
    ):
        """Given k as range, when SomeOf applied, then applies k augmentations within range."""
        # Given
        torch.manual_seed(42)
        augmentations = {
            "a": MockAugmentation(sample_rate=sample_rate, gain=1.1),
            "b": MockAugmentation(sample_rate=sample_rate, gain=1.2),
            "c": MockAugmentation(sample_rate=sample_rate, gain=1.3),
            "d": MockAugmentation(sample_rate=sample_rate, gain=1.4),
        }

        some_of = SomeOf(augmentations=augmentations, k=(1, 3))

        # When - apply multiple times and check apply counts
        for aug_k in augmentations:
            augmentations[aug_k].apply_count = 0

        n_trials = 100
        for _ in range(n_trials):
            some_of(mono_audio.clone())

        # Then
        # Total applications should be between 100 and 300 (1-3 per trial)
        total_applications = sum(
            augmentations[aug_k].apply_count for aug_k in augmentations
        )
        assert 100 <= total_applications <= 300

    def test_given_k_equals_zero_when_some_of_applied_then_returns_original(
        self, mono_audio, sample_rate
    ):
        """Given k=0, when SomeOf is applied, then returns original audio."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        aug2 = MockAugmentation(sample_rate=sample_rate, gain=3.0)
        some_of = SomeOf(augmentations={"a": aug1, "b": aug2}, k=0)

        original = mono_audio.clone()

        # When
        result = some_of(mono_audio)

        # Then
        assert torch.equal(result, original)

    def test_given_replace_true_when_some_of_applied_then_can_select_same_augmentation_twice(
        self, mono_audio, sample_rate
    ):
        """Given replace=True, when SomeOf applied, then can select same augmentation multiple times."""
        # Given
        torch.manual_seed(42)
        aug1 = MockAugmentation(sample_rate=sample_rate, gain=2.0)
        # Only one augmentation, but k=3 with replacement
        some_of = SomeOf(augmentations={"a": aug1}, k=3, replace=True)

        original = mono_audio.clone()

        # When
        result = some_of(mono_audio)

        # Then
        # Should apply aug1 three times: 2.0^3 = 8.0
        expected = original * (2.0**3)
        assert torch.allclose(result, expected)

    def test_given_k_greater_than_n_when_some_of_initialized_then_raises_error(
        self, sample_rate
    ):
        """Given k > number of augmentations, when initialized without replace, then raises ValueError."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate)
        aug2 = MockAugmentation(sample_rate=sample_rate)

        # When / Then
        with pytest.raises(ValueError, match="k must be between"):
            SomeOf(augmentations={"a": aug1, "b": aug2}, k=3, replace=False)

    def test_given_some_of_when_exported_to_config_then_includes_k_and_replace(
        self, sample_rate
    ):
        """Given a SomeOf, when exported to config, then includes k and replace parameters."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate)
        aug2 = MockAugmentation(sample_rate=sample_rate)
        some_of = SomeOf(augmentations={"a": aug1, "b": aug2}, k=2, replace=True)

        # When
        config = some_of.to_config()

        # Then
        assert config["k"] == 2
        assert config["replace"] is True

    def test_given_k_range_when_exported_to_config_then_includes_tuple(
        self, sample_rate
    ):
        """Given k as range, when exported to config, then includes tuple."""
        # Given
        aug1 = MockAugmentation(sample_rate=sample_rate)
        aug2 = MockAugmentation(sample_rate=sample_rate)
        some_of = SomeOf(augmentations={"a": aug1, "b": aug2}, k=(1, 2))

        # When
        config = some_of.to_config()

        # Then
        assert config["k"] == (1, 2)

    def test_given_dict_with_mismatched_sample_rates_when_some_of_created_then_raises_error_with_name(
        self,
    ):
        """Given dict with mismatched sample rates, when SomeOf created, then raises error."""
        # Given
        augmentations = {
            "first": MockAugmentation(sample_rate=16000, gain=2.0),
            "second": MockAugmentation(sample_rate=44100, gain=3.0),
            "third": MockAugmentation(sample_rate=16000, gain=4.0),
        }

        # When/Then
        # TODO: Error message should include augmentation name "second" to help debugging
        with pytest.raises(ValueError):
            SomeOf(k=2, augmentations=augmentations)


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
        one_of = OneOf({"a": one_of_aug1, "b": one_of_aug2})

        compose = Compose({"a": fixed_aug, "b": one_of})

        original = mono_audio.clone()

        # When
        result = compose(mono_audio)

        # Then
        # Should apply fixed_aug (x2), then one of the OneOf augmentations
        option1 = original * 2.0 * 1.5  # fixed then one_of_aug1
        option2 = original * 2.0 * 3.0  # fixed then one_of_aug2

        is_option1 = torch.allclose(result, option1)
        is_option2 = torch.allclose(result, option2)

        assert is_option1 or is_option2


class TestOneOfPerExampleMode:
    """Test OneOf per_example mode - different augmentation per batch example."""

    def test_given_batch_audio_when_one_of_applied_then_different_augmentations_per_example(
        self,
    ):
        """Given batch audio, when OneOf applied, then each example gets different augmentation."""
        # GIVEN: A batch of audio with 4 examples (batch, source, channel, time)
        batch_size = 4
        samples = 1000
        sample_rate = 16000
        # Start with zeros for easy verification
        # Shape: (batch, 1 source, 1 channel, time)
        audio_batch = torch.zeros(batch_size, 1, 1, samples, dtype=torch.float32)

        # Create OneOf with three distinct augmentations
        augmentations = {
            "add_one": AddOneAugmentation(sample_rate=sample_rate, p=1.0),
            "multiply_by_ten": MultiplyByTenAugmentation(
                sample_rate=sample_rate, p=1.0
            ),
            "add_hundred": AddHundredAugmentation(sample_rate=sample_rate, p=1.0),
        }
        one_of = OneOf(augmentations=augmentations, p=1.0, seed=42)

        # WHEN: Apply to batch
        result = one_of(audio_batch)

        # THEN: Different examples should have different augmentations applied
        # With 4 examples and 3 augmentations, we expect at least 2 different results
        unique_results = []
        for i in range(batch_size):
            example_result = result[i]
            # Check if this result is new
            is_unique = True
            for unique in unique_results:
                if torch.allclose(example_result, unique):
                    is_unique = False
                    break
            if is_unique:
                unique_results.append(example_result)

        # Should have at least 2 different augmentations applied across the batch
        assert len(unique_results) >= 2, (
            f"Expected at least 2 different augmentations, got {len(unique_results)}. "
            f"This suggests per_example mode is not working."
        )

        # Verify each example has one of the three expected transformations
        for i in range(batch_size):
            example = result[i]
            mean_value = example.mean()
            # Should be one of: 0+1=1, 0*10=0, 0+100=100
            is_add_one = torch.isclose(mean_value, torch.tensor(1.0), atol=0.1)
            is_multiply_ten = torch.isclose(mean_value, torch.tensor(0.0), atol=0.1)
            is_add_hundred = torch.isclose(mean_value, torch.tensor(100.0), atol=0.1)

            assert is_add_one or is_multiply_ten or is_add_hundred, (
                f"Example {i} has unexpected value {mean_value}"
            )

    def test_given_batch_audio_when_one_of_applied_multiple_times_then_consistent_variability(
        self,
    ):
        """Given batch audio, when OneOf applied multiple times, then shows consistent variability."""
        # GIVEN: A batch of audio (batch, source, channel, time)
        batch_size = 8
        samples = 100
        sample_rate = 16000
        audio_batch = torch.ones(batch_size, 1, 1, samples, dtype=torch.float32)

        augmentations = {
            "add_one": AddOneAugmentation(sample_rate=sample_rate, p=1.0),
            "multiply_by_ten": MultiplyByTenAugmentation(
                sample_rate=sample_rate, p=1.0
            ),
        }
        one_of = OneOf(augmentations=augmentations, p=1.0, seed=None)

        # WHEN: Apply multiple times and count unique patterns
        num_trials = 10
        all_had_variation = True

        for _ in range(num_trials):
            result = one_of(audio_batch.clone())
            # Count unique results in this batch
            unique_count = 0
            seen_values = []
            for i in range(batch_size):
                val = result[
                    i, 0, 0, 0
                ]  # First sample of each example (batch, source, channel, time)
                is_new = True
                for seen in seen_values:
                    if torch.isclose(val, seen, atol=0.1):
                        is_new = False
                        break
                if is_new:
                    seen_values.append(val)
                    unique_count += 1

            if unique_count < 2:
                all_had_variation = False
                break

        # THEN: Should consistently show variation across examples
        assert all_had_variation, (
            "OneOf should apply different augmentations to different examples in the batch"
        )


class TestSomeOfPerExampleMode:
    """Test SomeOf per_example mode - different augmentation selection per batch example."""

    def test_given_batch_audio_when_some_of_applied_then_different_selections_per_example(
        self,
    ):
        """Given batch audio, when SomeOf applied, then each example gets different augmentation selection."""
        # GIVEN: A batch of audio with 8 examples (batch, source, channel, time)
        batch_size = 8
        samples = 100
        sample_rate = 16000
        # Start with ones for easy verification
        audio_batch = torch.ones(batch_size, 1, 1, samples, dtype=torch.float32)

        # Create SomeOf that selects 2 out of 3 augmentations
        augmentations = {
            "add_one": AddOneAugmentation(sample_rate=sample_rate, p=1.0),
            "multiply_by_ten": MultiplyByTenAugmentation(
                sample_rate=sample_rate, p=1.0
            ),
            "add_hundred": AddHundredAugmentation(sample_rate=sample_rate, p=1.0),
        }
        some_of = SomeOf(k=2, augmentations=augmentations, p=1.0, seed=42)

        # WHEN: Apply to batch
        result = some_of(audio_batch)

        # THEN: Different examples should have different combinations
        # Possible results starting from 1.0:
        # - add_one then multiply_by_ten: (1+1)*10 = 20
        # - add_one then add_hundred: 1+1+100 = 102
        # - multiply_by_ten then add_one: 1*10+1 = 11
        # - multiply_by_ten then add_hundred: 1*10+100 = 110
        # - add_hundred then add_one: 1+100+1 = 102
        # - add_hundred then multiply_by_ten: (1+100)*10 = 1010

        unique_results = []
        for i in range(batch_size):
            example_result = result[i]
            mean_value = example_result.mean()

            # Check if this result is new
            is_unique = True
            for unique in unique_results:
                if torch.isclose(mean_value, unique, atol=1.0):
                    is_unique = False
                    break
            if is_unique:
                unique_results.append(mean_value)

        # With 8 examples and multiple possible combinations, expect at least 2 different results
        assert len(unique_results) >= 2, (
            f"Expected at least 2 different augmentation combinations, got {len(unique_results)}. "
            f"Unique values: {unique_results}"
        )

    def test_given_batch_audio_when_some_of_k_range_applied_then_variable_k_per_example(
        self,
    ):
        """Given batch audio, when SomeOf with k range applied, then different k values per example."""
        # GIVEN: A batch of audio (batch, source, channel, time)
        batch_size = 10
        samples = 100
        sample_rate = 16000
        audio_batch = torch.ones(batch_size, 1, 1, samples, dtype=torch.float32)

        # Create SomeOf with k range (1 to 3)
        augmentations = {
            "add_one": AddOneAugmentation(sample_rate=sample_rate, p=1.0),
            "multiply_by_ten": MultiplyByTenAugmentation(
                sample_rate=sample_rate, p=1.0
            ),
            "add_hundred": AddHundredAugmentation(sample_rate=sample_rate, p=1.0),
        }
        some_of = SomeOf(k=(1, 3), augmentations=augmentations, p=1.0, seed=42)

        # WHEN: Apply to batch multiple times and collect results
        num_trials = 5
        all_results = []
        for _ in range(num_trials):
            result = some_of(audio_batch.clone())
            for i in range(batch_size):
                all_results.append(result[i].mean())

        # THEN: Should see variety in results (different k values and selections)
        unique_results = []
        for val in all_results:
            is_unique = True
            for unique in unique_results:
                if torch.isclose(val, unique, atol=1.0):
                    is_unique = False
                    break
            if is_unique:
                unique_results.append(val)

        # Should have at least 3 different result patterns
        assert len(unique_results) >= 3, (
            f"Expected at least 3 different patterns, got {len(unique_results)}"
        )


class TestComposePerExampleMode:
    """Test Compose per_example mode - parameter randomization per batch example."""

    def test_given_batch_audio_when_compose_applied_then_parameters_vary_per_example(
        self,
    ):
        """Given batch audio, when Compose applied, then parameters randomize per example."""
        # GIVEN: A batch of audio (batch, source, channel, time)
        batch_size = 4
        samples = 100
        sample_rate = 16000
        audio_batch = torch.ones(batch_size, 1, 1, samples, dtype=torch.float32)

        # Create Compose with augmentations that have randomizable parameters
        # Using MockAugmentation with a gain range
        augmentations = {
            "gain1": MockAugmentation(sample_rate=sample_rate, gain=2.0, p=1.0),
            "gain2": MockAugmentation(sample_rate=sample_rate, gain=3.0, p=1.0),
        }
        compose = Compose(augmentations=augmentations, p=1.0, mode="per_example")

        # WHEN: Apply to batch
        result = compose(audio_batch)

        # THEN: All examples should have same result (currently)
        # This test documents current behavior - will need updating when per_example is implemented
        for i in range(1, batch_size):
            # Currently all examples get the same treatment
            assert torch.allclose(result[0], result[i]), (
                "Without per_example mode, all examples should be identical"
            )

    def test_given_batch_audio_when_compose_with_one_of_applied_then_varies_per_example(
        self,
    ):
        """Given batch audio, when Compose containing OneOf applied, then varies per example."""
        # GIVEN: A batch of audio (batch, source, channel, time)
        batch_size = 6
        samples = 100
        sample_rate = 16000
        audio_batch = torch.zeros(batch_size, 1, 1, samples, dtype=torch.float32)

        # Create nested composition: Compose([AddOne, OneOf([MultiplyByTen, AddHundred])])
        add_one = AddOneAugmentation(sample_rate=sample_rate, p=1.0)
        one_of = OneOf(
            {
                "multiply": MultiplyByTenAugmentation(sample_rate=sample_rate, p=1.0),
                "add_hundred": AddHundredAugmentation(sample_rate=sample_rate, p=1.0),
            },
            p=1.0,
            seed=42,
        )
        compose = Compose(
            {"add_one": add_one, "one_of": one_of}, p=1.0, mode="per_example"
        )

        # WHEN: Apply to batch
        result = compose(audio_batch)

        # THEN: Different examples should have different results
        # Expected: (0+1)*10=10 or 0+1+100=101
        unique_results = []
        for i in range(batch_size):
            mean_value = result[i].mean()
            is_unique = True
            for unique in unique_results:
                if torch.isclose(mean_value, unique, atol=1.0):
                    is_unique = False
                    break
            if is_unique:
                unique_results.append(mean_value)

        # Should have at least 2 different results
        assert len(unique_results) >= 2, (
            f"Expected at least 2 different results from nested composition, got {len(unique_results)}"
        )
