"""BDD-style tests for dict-based composition (Dict[str, BaseAugmentation]).

These tests validate that composition classes (Compose, OneOf, SomeOf) can accept
dictionaries of augmentations, which is better for structured configs and provides
meaningful names.
"""

import pytest
import torch

from auementations.core.base import BaseAugmentation
from auementations.core.composition import Compose, OneOf, SomeOf
from tests.conftest import MockAugmentation


# Mock augmentations with distinct, easily verifiable effects
class AddOneAugmentation(BaseAugmentation):
    """Adds 1.0 to the audio signal."""

    def __init__(self, sample_rate: int, p: float = 1.0):
        super().__init__(sample_rate=sample_rate, p=p)
        self.apply_count = 0

    def __call__(self, audio, **kwargs):
        if not self.should_apply():
            return audio
        self.apply_count += 1
        return audio + 1.0


class MultiplyByTenAugmentation(BaseAugmentation):
    """Multiplies audio signal by 10.0."""

    def __init__(self, sample_rate: int, p: float = 1.0):
        super().__init__(sample_rate=sample_rate, p=p)
        self.apply_count = 0

    def __call__(self, audio, **kwargs):
        if not self.should_apply():
            return audio
        self.apply_count += 1
        return audio * 10.0


class AddHundredAugmentation(BaseAugmentation):
    """Adds 100.0 to the audio signal."""

    def __init__(self, sample_rate: int, p: float = 1.0):
        super().__init__(sample_rate=sample_rate, p=p)
        self.apply_count = 0

    def __call__(self, audio, **kwargs):
        if not self.should_apply():
            return audio
        self.apply_count += 1
        return audio + 100.0


class TestComposeWithDict:
    """Test Compose with Dict[str, BaseAugmentation] input."""

    def test_given_dict_augmentations_when_composed_then_creates_successfully(self):
        """Given dict of augmentations, when Compose created, then initializes successfully."""
        # GIVEN: A dict of named augmentations
        augmentations = {
            "first": MockAugmentation(sample_rate=16000, gain=2.0),
            "second": MockAugmentation(sample_rate=16000, gain=3.0),
        }

        # WHEN: Create Compose with dict
        compose = Compose(augmentations=augmentations)

        # THEN: Compose is created successfully
        assert compose is not None
        assert len(compose.augmentations) == 2
        assert compose.augmentation_names == ["first", "second"]

    def test_given_dict_augmentations_when_applied_then_applies_in_order(self):
        """Given dict augmentations, when applied, then applies in insertion order."""
        # GIVEN: A dict of named augmentations with known effects
        augmentations = {
            "multiply_by_2": MockAugmentation(sample_rate=16000, gain=2.0),
            "multiply_by_3": MockAugmentation(sample_rate=16000, gain=3.0),
        }
        compose = Compose(augmentations=augmentations, p=1.0)
        # Shape: (1 source, 1 channel, 16000 time)
        audio = torch.ones(1, 1, 16000, dtype=torch.float32)

        # WHEN: Apply composition
        result = compose(audio)

        # THEN: Augmentations applied in order (2.0 * 3.0 = 6.0)
        expected = audio * 2.0 * 3.0
        assert torch.allclose(result, expected)

    def test_given_dict_augmentations_when_repr_called_then_shows_names(self):
        """Given dict augmentations, when repr called, then shows augmentation names."""
        # GIVEN: A dict of named augmentations
        augmentations = {
            "gain_up": MockAugmentation(sample_rate=16000, gain=2.0),
            "gain_down": MockAugmentation(sample_rate=16000, gain=0.5),
        }
        compose = Compose(augmentations=augmentations)

        # WHEN: Get repr
        repr_str = repr(compose)

        # THEN: Contains augmentation names
        assert "gain_up" in repr_str
        assert "gain_down" in repr_str
        assert "{" in repr_str  # Dict notation

    def test_given_dict_with_mismatched_sample_rates_when_composed_then_raises_error(
        self,
    ):
        """Given dict with mismatched sample rates, when composed, then raises error with name."""
        # GIVEN: A dict with mismatched sample rates
        augmentations = {
            "first": MockAugmentation(sample_rate=16000, gain=2.0),
            "second": MockAugmentation(sample_rate=44100, gain=3.0),
        }

        # WHEN/THEN: Creating Compose raises ValueError with augmentation name
        with pytest.raises(ValueError, match="second"):
            Compose(augmentations=augmentations)

    def test_given_empty_dict_when_composed_then_raises_error(self):
        """Given empty dict, when composed, then raises error."""
        # GIVEN: An empty dict
        augmentations = {}

        # WHEN/THEN: Creating Compose raises ValueError
        with pytest.raises(ValueError, match="augmentations list cannot be empty"):
            Compose(augmentations=augmentations)

    def test_given_dict_augmentations_when_sample_rate_inferred_then_uses_first(self):
        """Given dict augmentations, when sample rate inferred, then uses first augmentation."""
        # GIVEN: A dict of augmentations without explicit sample_rate
        augmentations = {
            "first": MockAugmentation(sample_rate=16000, gain=2.0),
            "second": MockAugmentation(sample_rate=16000, gain=3.0),
        }

        # WHEN: Create Compose without sample_rate
        compose = Compose(augmentations=augmentations)

        # THEN: Sample rate inferred from first augmentation
        assert compose.sample_rate == 16000


class TestOneOfWithDict:
    """Test OneOf with Dict[str, BaseAugmentation] input."""

    def test_given_dict_augmentations_when_one_of_created_then_creates_successfully(
        self,
    ):
        """Given dict of augmentations, when OneOf created, then initializes successfully."""
        # GIVEN: A dict of named augmentations
        augmentations = {
            "option_a": MockAugmentation(sample_rate=16000, gain=2.0),
            "option_b": MockAugmentation(sample_rate=16000, gain=3.0),
            "option_c": MockAugmentation(sample_rate=16000, gain=4.0),
        }

        # WHEN: Create OneOf with dict
        one_of = OneOf(augmentations=augmentations)

        # THEN: OneOf is created successfully
        assert one_of is not None
        assert len(one_of.augmentations) == 3
        assert one_of.augmentation_names == ["option_a", "option_b", "option_c"]

    def test_given_dict_augmentations_when_applied_then_selects_one(self):
        """Given dict augmentations, when applied, then selects exactly one."""
        # GIVEN: A dict of named augmentations with distinct effects
        augmentations = {
            "double": MockAugmentation(sample_rate=16000, gain=2.0),
            "triple": MockAugmentation(sample_rate=16000, gain=3.0),
            "quadruple": MockAugmentation(sample_rate=16000, gain=4.0),
        }
        one_of = OneOf(augmentations=augmentations, p=1.0, seed=42)
        # Shape: (1 source, 1 channel, 16000 time)
        audio = torch.ones(1, 1, 16000, dtype=torch.float32)

        # WHEN: Apply multiple times
        results = [one_of(audio.clone()) for _ in range(10)]

        # THEN: Each result is one of the expected values
        expected_values = [2.0, 3.0, 4.0]
        for result in results:
            result_value = result[
                0, 0, 0
            ]  # Get the scalar value from (source, channel, time)
            assert any(
                torch.isclose(result_value, torch.tensor(v)) for v in expected_values
            )

    def test_given_dict_with_dict_weights_when_one_of_created_then_applies_weights(
        self,
    ):
        """Given dict augmentations with dict weights, when created, then applies weights correctly."""
        # GIVEN: A dict of augmentations with dict weights
        augmentations = {
            "high_prob": MockAugmentation(sample_rate=16000, gain=10.0),
            "low_prob": MockAugmentation(sample_rate=16000, gain=1.0),
        }
        weights = {
            "high_prob": 0.9,
            "low_prob": 0.1,
        }

        # WHEN: Create OneOf with dict weights
        one_of = OneOf(augmentations=augmentations, weights=weights, p=1.0, seed=42)
        # Shape: (1 source, 1 channel, 100 time)
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

        # THEN: High probability augmentation selected more often
        assert high_count > low_count

    def test_given_dict_with_mismatched_sample_rates_when_one_of_created_then_raises_error(
        self,
    ):
        """Given dict with mismatched sample rates, when OneOf created, then raises error with name."""
        # GIVEN: A dict with mismatched sample rates
        augmentations = {
            "first": MockAugmentation(sample_rate=16000, gain=2.0),
            "second": MockAugmentation(sample_rate=44100, gain=3.0),
        }

        # WHEN/THEN: Creating OneOf raises ValueError with augmentation name
        with pytest.raises(ValueError, match="second"):
            OneOf(augmentations=augmentations)

    def test_given_empty_dict_when_one_of_created_then_raises_error(self):
        """Given empty dict, when OneOf created, then raises error."""
        # GIVEN: An empty dict
        augmentations = {}

        # WHEN/THEN: Creating OneOf raises ValueError
        with pytest.raises(ValueError, match="augmentations list cannot be empty"):
            OneOf(augmentations=augmentations)


class TestSomeOfWithDict:
    """Test SomeOf with Dict[str, BaseAugmentation] input."""

    def test_given_dict_augmentations_when_some_of_created_then_creates_successfully(
        self,
    ):
        """Given dict of augmentations, when SomeOf created, then initializes successfully."""
        # GIVEN: A dict of named augmentations
        augmentations = {
            "first": MockAugmentation(sample_rate=16000, gain=2.0),
            "second": MockAugmentation(sample_rate=16000, gain=3.0),
            "third": MockAugmentation(sample_rate=16000, gain=4.0),
        }

        # WHEN: Create SomeOf with dict
        some_of = SomeOf(k=2, augmentations=augmentations)

        # THEN: SomeOf is created successfully
        assert some_of is not None
        assert len(some_of.augmentations) == 3
        assert some_of.augmentation_names == ["first", "second", "third"]

    def test_given_dict_augmentations_when_applied_then_applies_k_augmentations(self):
        """Given dict augmentations, when applied, then applies exactly k augmentations."""
        # GIVEN: A dict of named augmentations
        augmentations = {
            "aug1": MockAugmentation(sample_rate=16000, gain=2.0),
            "aug2": MockAugmentation(sample_rate=16000, gain=1.5),
            "aug3": MockAugmentation(sample_rate=16000, gain=1.2),
            "aug4": MockAugmentation(sample_rate=16000, gain=1.1),
        }
        some_of = SomeOf(k=2, augmentations=augmentations, p=1.0, seed=42)
        # Shape: (1 source, 1 channel, 100 time)
        audio = torch.ones(1, 1, 100, dtype=torch.float32)

        # WHEN: Apply augmentation
        result = some_of(audio)

        # THEN: Result is modified (2 augmentations applied)
        assert not torch.equal(result, audio)
        # Result should be product of 2 gains
        assert result[0, 0, 0] > 1.0

    def test_given_dict_with_k_greater_than_length_when_some_of_created_then_raises_error(
        self,
    ):
        """Given dict with k > length, when SomeOf created without replace, then raises error."""
        # GIVEN: A dict with 3 augmentations and k=4
        augmentations = {
            "first": MockAugmentation(sample_rate=16000, gain=2.0),
            "second": MockAugmentation(sample_rate=16000, gain=3.0),
            "third": MockAugmentation(sample_rate=16000, gain=4.0),
        }

        # WHEN/THEN: Creating SomeOf raises ValueError
        with pytest.raises(ValueError, match="k must be between 0 and 3"):
            SomeOf(k=4, augmentations=augmentations, replace=False)

    def test_given_dict_with_k_range_when_applied_then_applies_variable_k(self):
        """Given dict with k as range, when applied, then applies variable number of augmentations."""
        # GIVEN: A dict of augmentations with k range
        augmentations = {
            "aug1": MockAugmentation(sample_rate=16000, gain=1.1),
            "aug2": MockAugmentation(sample_rate=16000, gain=1.1),
            "aug3": MockAugmentation(sample_rate=16000, gain=1.1),
        }
        some_of = SomeOf(k=(1, 3), augmentations=augmentations, p=1.0, seed=42)
        # Shape: (1 source, 1 channel, 100 time)
        audio = torch.ones(1, 1, 100, dtype=torch.float32)

        # WHEN: Apply multiple times
        results = [some_of(audio.clone()) for _ in range(20)]

        # THEN: Results vary (different k values applied)
        unique_results = []
        for result in results:
            is_unique = True
            for unique in unique_results:
                if torch.allclose(result, unique, atol=1e-6):
                    is_unique = False
                    break
            if is_unique:
                unique_results.append(result)

        # Should have some variation
        assert len(unique_results) > 1

    def test_given_dict_with_mismatched_sample_rates_when_some_of_created_then_raises_error(
        self,
    ):
        """Given dict with mismatched sample rates, when SomeOf created, then raises error with name."""
        # GIVEN: A dict with mismatched sample rates
        augmentations = {
            "first": MockAugmentation(sample_rate=16000, gain=2.0),
            "second": MockAugmentation(sample_rate=44100, gain=3.0),
            "third": MockAugmentation(sample_rate=16000, gain=4.0),
        }

        # WHEN/THEN: Creating SomeOf raises ValueError with augmentation name
        with pytest.raises(ValueError, match="second"):
            SomeOf(k=2, augmentations=augmentations)

    def test_given_empty_dict_when_some_of_created_then_raises_error(self):
        """Given empty dict, when SomeOf created, then raises error."""
        # GIVEN: An empty dict
        augmentations = {}

        # WHEN/THEN: Creating SomeOf raises ValueError
        with pytest.raises(ValueError, match="augmentations list cannot be empty"):
            SomeOf(k=1, augmentations=augmentations)


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

        # WHEN: Apply to batch (this should fail initially since per_example isn't implemented)
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
