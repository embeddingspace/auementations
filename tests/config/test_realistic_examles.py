"""BDD-style tests for realistic usage examples with full configs and composition.

These tests demonstrate real-world usage patterns combining:
- Hydra/hydra-zen configuration
- Composition patterns (OneOf, Compose, etc.)
- Multiple augmentation backends (torch_audiomentations, pedalboard)
- Stochastic parameter sampling
"""

import numpy as np
import pytest

pytest.importorskip("pedalboard")

from hydra_zen import builds, instantiate

from auementations.config.config_store import auementations_store
from auementations.core.composition import OneOf


class TestRealisticOneOfComposition:
    """Test realistic usage of OneOf composition with configs."""

    def test_given_one_of_config_when_instantiated_then_creates_composition(
        self, mono_audio
    ):
        """Given OneOf config with LPF/HPF, when instantiated, then creates working composition."""
        # GIVEN: A config that uses OneOf to randomly select between LPF and HPF
        sample_rate = 16000

        # Get the registered configs from the store
        lpf_config = auementations_store.get_entry(group="pedalboard", name="lpf")[
            "node"
        ]
        hpf_config = auementations_store.get_entry(group="pedalboard", name="hpf")[
            "node"
        ]
        one_of_config = auementations_store.get_entry(
            group="composition", name="one_of"
        )["node"]

        # Build a complete config for OneOf composition
        # This creates a structured config that can be instantiated by Hydra
        composition_config = builds(
            one_of_config,
            augmentations=[
                builds(
                    lpf_config,
                    sample_rate=sample_rate,
                    min_cutoff_freq=500.0,
                    max_cutoff_freq=2000.0,
                ),
                builds(
                    hpf_config,
                    sample_rate=sample_rate,
                    min_cutoff_freq=100.0,
                    max_cutoff_freq=500.0,
                ),
            ],
            sample_rate=sample_rate,
            p=1.0,
        )

        # WHEN: Instantiate the config to create the actual augmentation pipeline
        augmentation = instantiate(composition_config)

        # THEN: The instantiated object is a OneOf composition
        assert isinstance(augmentation, OneOf)
        assert augmentation.sample_rate == sample_rate
        assert len(augmentation.augmentations) == 2

        # AND: The composition can process audio successfully
        result = augmentation(mono_audio)
        assert result.shape == mono_audio.shape
        assert isinstance(result, np.ndarray)

    def test_given_one_of_composition_when_applied_multiple_times_then_produces_different_results(
        self, mono_audio
    ):
        """Given OneOf composition, when applied multiple times, then produces different results due to randomness."""
        # GIVEN: A OneOf composition with distinct filters
        sample_rate = 16000

        lpf_config = auementations_store.get_entry(group="pedalboard", name="lpf")[
            "node"
        ]
        hpf_config = auementations_store.get_entry(group="pedalboard", name="hpf")[
            "node"
        ]
        one_of_config = auementations_store.get_entry(
            group="composition", name="one_of"
        )["node"]

        composition_config = builds(
            one_of_config,
            augmentations=[
                builds(
                    lpf_config,
                    sample_rate=sample_rate,
                    cutoff_freq=800.0,  # Fixed LPF at 800Hz
                ),
                builds(
                    hpf_config,
                    sample_rate=sample_rate,
                    cutoff_freq=3000.0,  # Fixed HPF at 3000Hz
                ),
            ],
            sample_rate=sample_rate,
            p=1.0,
            seed=None,  # No seed for true randomness
        )

        augmentation = instantiate(composition_config)

        # WHEN: Apply the augmentation multiple times
        num_trials = 20
        results = []
        for _ in range(num_trials):
            result = augmentation(mono_audio.copy())
            results.append(result)

        # THEN: At least some results should be different
        # (Since OneOf randomly picks between LPF and HPF, we should get variation)
        # Compare results pairwise - not all should be identical
        different_count = 0
        for i in range(num_trials - 1):
            if not np.allclose(results[i], results[i + 1], atol=1e-6):
                different_count += 1

        # We should see at least some variation (not all pairs identical)
        assert different_count > 0, "All results were identical - OneOf not randomizing"

    def test_given_one_of_with_range_parameters_when_applied_then_samples_from_ranges(
        self, mono_audio
    ):
        """Given OneOf with range parameters, when applied, then samples different values from ranges."""
        # GIVEN: A OneOf composition where filters have parameter ranges
        sample_rate = 16000

        lpf_config = auementations_store.get_entry(group="pedalboard", name="lpf")[
            "node"
        ]
        hpf_config = auementations_store.get_entry(group="pedalboard", name="hpf")[
            "node"
        ]
        one_of_config = auementations_store.get_entry(
            group="composition", name="one_of"
        )["node"]

        composition_config = builds(
            one_of_config,
            augmentations=[
                builds(
                    lpf_config,
                    sample_rate=sample_rate,
                    min_cutoff_freq=500.0,
                    max_cutoff_freq=2000.0,
                ),
                builds(
                    hpf_config,
                    sample_rate=sample_rate,
                    min_cutoff_freq=100.0,
                    max_cutoff_freq=1000.0,
                ),
            ],
            sample_rate=sample_rate,
            p=1.0,
        )

        augmentation = instantiate(composition_config)

        # WHEN: Apply multiple times
        num_trials = 10
        results = []
        for _ in range(num_trials):
            result = augmentation(mono_audio.copy())
            results.append(result)

        # THEN: Results should vary because:
        # 1. OneOf randomly selects which filter
        # 2. Each filter randomly samples cutoff from its range
        unique_results = []
        for result in results:
            is_unique = True
            for unique in unique_results:
                if np.allclose(result, unique, atol=1e-6):
                    is_unique = False
                    break
            if is_unique:
                unique_results.append(result)

        # Should have multiple unique results due to randomness
        assert len(unique_results) > 1, (
            "No variation in results despite range parameters"
        )

    def test_given_dict_config_when_instantiated_then_creates_working_augmentation(
        self, mono_audio
    ):
        """Given a dictionary-based config, when instantiated, then creates working augmentation."""
        # GIVEN: A config specified as a dictionary (like from a YAML file)
        sample_rate = 16000

        # This mimics what you'd get from a Hydra YAML config file
        config_dict = {
            "_target_": "auementations.core.composition.OneOf",
            "augmentations": [
                {
                    "_target_": "auementations.adapters.pedalboard.LowPassFilter",
                    "sample_rate": sample_rate,
                    "min_cutoff_freq": 500.0,
                    "max_cutoff_freq": 2000.0,
                    "p": 1.0,
                },
                {
                    "_target_": "auementations.adapters.pedalboard.HighPassFilter",
                    "sample_rate": sample_rate,
                    "min_cutoff_freq": 100.0,
                    "max_cutoff_freq": 500.0,
                    "p": 1.0,
                },
            ],
            "sample_rate": sample_rate,
            "p": 1.0,
        }

        # WHEN: Instantiate from the dictionary config
        augmentation = instantiate(config_dict)

        # THEN: Creates a working OneOf composition
        assert isinstance(augmentation, OneOf)
        assert augmentation.sample_rate == sample_rate

        # AND: Can process audio
        result = augmentation(mono_audio)
        assert result.shape == mono_audio.shape

        # AND: Multiple applications produce variation
        results = [augmentation(mono_audio.copy()) for _ in range(10)]
        # Check that not all results are identical
        all_same = all(np.allclose(results[0], r, atol=1e-6) for r in results[1:])
        assert not all_same, "All results identical - composition not working correctly"
