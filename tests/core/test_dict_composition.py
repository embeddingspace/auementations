"""BDD-style tests for dict-based composition (Dict[str, BaseAugmentation]).

These tests validate that composition classes (Compose, OneOf, SomeOf) can accept
dictionaries of augmentations, which is better for structured configs and provides
meaningful names.
"""

import numpy as np
import pytest

from auementations.core.composition import Compose, OneOf, SomeOf
from tests.conftest import MockAugmentation


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
        audio = np.ones(16000, dtype=np.float32)

        # WHEN: Apply composition
        result = compose(audio)

        # THEN: Augmentations applied in order (2.0 * 3.0 = 6.0)
        expected = audio * 2.0 * 3.0
        assert np.allclose(result, expected)

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
        audio = np.ones(16000, dtype=np.float32)

        # WHEN: Apply multiple times
        results = [one_of(audio.copy()) for _ in range(10)]

        # THEN: Each result is one of the expected values
        expected_values = [2.0, 3.0, 4.0]
        for result in results:
            result_value = result[0]  # Get the scalar value
            assert any(np.isclose(result_value, v) for v in expected_values)

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
        audio = np.ones(100, dtype=np.float32)

        # Apply many times and count
        high_count = 0
        low_count = 0
        for _ in range(100):
            result = one_of(audio.copy())
            if np.isclose(result[0], 10.0):
                high_count += 1
            else:
                low_count += 1

        # THEN: High probability augmentation selected more often
        assert high_count > low_count

    def test_given_dict_weights_with_list_augmentations_when_created_then_raises_error(
        self,
    ):
        """Given dict weights with list augmentations, when created, then raises error."""
        # GIVEN: List augmentations with dict weights
        augmentations = [
            MockAugmentation(sample_rate=16000, gain=2.0),
            MockAugmentation(sample_rate=16000, gain=3.0),
        ]
        weights = {"first": 0.5, "second": 0.5}

        # WHEN/THEN: Creating OneOf raises ValueError
        with pytest.raises(ValueError, match="Cannot use dict weights"):
            OneOf(augmentations=augmentations, weights=weights)

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
        audio = np.ones(100, dtype=np.float32)

        # WHEN: Apply augmentation
        result = some_of(audio)

        # THEN: Result is modified (2 augmentations applied)
        assert not np.array_equal(result, audio)
        # Result should be product of 2 gains
        assert result[0] > 1.0

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
        audio = np.ones(100, dtype=np.float32)

        # WHEN: Apply multiple times
        results = [some_of(audio.copy()) for _ in range(20)]

        # THEN: Results vary (different k values applied)
        unique_results = []
        for result in results:
            is_unique = True
            for unique in unique_results:
                if np.allclose(result, unique, atol=1e-6):
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


class TestDictCompositionBackwardCompatibility:
    """Test that dict-based composition is backward compatible with list-based."""

    def test_given_list_augmentations_when_composed_then_still_works(self):
        """Given list augmentations, when Compose created, then works as before."""
        # GIVEN: A list of augmentations (old API)
        augmentations = [
            MockAugmentation(sample_rate=16000, gain=2.0),
            MockAugmentation(sample_rate=16000, gain=3.0),
        ]

        # WHEN: Create Compose with list
        compose = Compose(augmentations=augmentations)
        audio = np.ones(100, dtype=np.float32)
        result = compose(audio)

        # THEN: Works correctly
        assert compose.augmentation_names is None  # No names for list
        assert np.allclose(result, audio * 2.0 * 3.0)

    def test_given_list_augmentations_when_one_of_created_then_still_works(self):
        """Given list augmentations, when OneOf created, then works as before."""
        # GIVEN: A list of augmentations (old API)
        augmentations = [
            MockAugmentation(sample_rate=16000, gain=2.0),
            MockAugmentation(sample_rate=16000, gain=3.0),
        ]

        # WHEN: Create OneOf with list
        one_of = OneOf(augmentations=augmentations)

        # THEN: Works correctly
        assert one_of.augmentation_names is None  # No names for list
        assert len(one_of.augmentations) == 2

    def test_given_list_augmentations_when_some_of_created_then_still_works(self):
        """Given list augmentations, when SomeOf created, then works as before."""
        # GIVEN: A list of augmentations (old API)
        augmentations = [
            MockAugmentation(sample_rate=16000, gain=2.0),
            MockAugmentation(sample_rate=16000, gain=3.0),
            MockAugmentation(sample_rate=16000, gain=4.0),
        ]

        # WHEN: Create SomeOf with list
        some_of = SomeOf(k=2, augmentations=augmentations)

        # THEN: Works correctly
        assert some_of.augmentation_names is None  # No names for list
        assert len(some_of.augmentations) == 3
