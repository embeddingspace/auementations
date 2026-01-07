"""BDD-style tests for realistic usage examples with full configs and composition.

These tests demonstrate real-world usage patterns combining:
- Hydra/hydra-zen configuration
- Composition patterns (OneOf, Compose, etc.)
- Multiple augmentation backends (torch_audiomentations, pedalboard)
- Stochastic parameter sampling
"""

import numpy as np
import pytest
import torch

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
        lpf_config = auementations_store.get_entry(
            group="auementations/pedalboard", name="lpf"
        )["node"]
        hpf_config = auementations_store.get_entry(
            group="auementations/pedalboard", name="hpf"
        )["node"]
        one_of_config = auementations_store.get_entry(
            group="auementations/composition", name="one_of"
        )["node"]

        # Build a complete config for OneOf composition
        # This creates a structured config that can be instantiated by Hydra
        composition_config = builds(
            one_of_config,
            augmentations={
                "lowpass": builds(
                    lpf_config,
                    sample_rate=sample_rate,
                    min_cutoff_freq=500.0,
                    max_cutoff_freq=2000.0,
                ),
                "highpass": builds(
                    hpf_config,
                    sample_rate=sample_rate,
                    min_cutoff_freq=100.0,
                    max_cutoff_freq=500.0,
                ),
            },
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
        assert isinstance(result, torch.Tensor)

    def test_given_one_of_composition_when_applied_multiple_times_then_produces_different_results(
        self, mono_audio
    ):
        """Given OneOf composition, when applied multiple times, then produces different results due to randomness."""
        # GIVEN: A OneOf composition with distinct filters
        sample_rate = 16000

        lpf_config = auementations_store.get_entry(
            group="auementations/pedalboard", name="lpf"
        )["node"]
        hpf_config = auementations_store.get_entry(
            group="auementations/pedalboard", name="hpf"
        )["node"]
        one_of_config = auementations_store.get_entry(
            group="auementations/composition", name="one_of"
        )["node"]

        composition_config = builds(
            one_of_config,
            augmentations={
                "lowpass": builds(
                    lpf_config,
                    sample_rate=sample_rate,
                    cutoff_freq=800.0,  # Fixed LPF at 800Hz
                ),
                "highpass": builds(
                    hpf_config,
                    sample_rate=sample_rate,
                    cutoff_freq=3000.0,  # Fixed HPF at 3000Hz
                ),
            },
            sample_rate=sample_rate,
            p=1.0,
            seed=None,  # No seed for true randomness
        )

        augmentation = instantiate(composition_config)

        # WHEN: Apply the augmentation multiple times
        num_trials = 20
        results = []
        for _ in range(num_trials):
            result = augmentation(mono_audio.clone())
            results.append(result)

        # THEN: At least some results should be different
        # (Since OneOf randomly picks between LPF and HPF, we should get variation)
        # Compare results pairwise - not all should be identical
        different_count = 0
        for i in range(num_trials - 1):
            if not torch.allclose(results[i], results[i + 1], atol=1e-6):
                different_count += 1

        # We should see at least some variation (not all pairs identical)
        assert different_count > 0, "All results were identical - OneOf not randomizing"

    def test_given_one_of_with_range_parameters_when_applied_then_samples_from_ranges(
        self, mono_audio
    ):
        """Given OneOf with range parameters, when applied, then samples different values from ranges."""
        # GIVEN: A OneOf composition where filters have parameter ranges
        sample_rate = 16000

        lpf_config = auementations_store.get_entry(
            group="auementations/pedalboard", name="lpf"
        )["node"]
        hpf_config = auementations_store.get_entry(
            group="auementations/pedalboard", name="hpf"
        )["node"]
        one_of_config = auementations_store.get_entry(
            group="auementations/composition", name="one_of"
        )["node"]

        composition_config = builds(
            one_of_config,
            augmentations={
                "lowpass": builds(
                    lpf_config,
                    sample_rate=sample_rate,
                    min_cutoff_freq=500.0,
                    max_cutoff_freq=2000.0,
                ),
                "highpass": builds(
                    hpf_config,
                    sample_rate=sample_rate,
                    min_cutoff_freq=100.0,
                    max_cutoff_freq=1000.0,
                ),
            },
            sample_rate=sample_rate,
            p=1.0,
        )

        augmentation = instantiate(composition_config)

        # WHEN: Apply multiple times
        num_trials = 10
        results = []
        for _ in range(num_trials):
            result = augmentation(mono_audio.clone())
            results.append(result)

        # THEN: Results should vary because:
        # 1. OneOf randomly selects which filter
        # 2. Each filter randomly samples cutoff from its range
        unique_results = []
        for result in results:
            is_unique = True
            for unique in unique_results:
                if torch.allclose(result, unique, atol=1e-6):
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
            "augmentations": {
                "lowpass": {
                    "_target_": "auementations.adapters.pedalboard.LowPassFilter",
                    "sample_rate": sample_rate,
                    "min_cutoff_freq": 500.0,
                    "max_cutoff_freq": 2000.0,
                    "p": 1.0,
                },
                "highpass": {
                    "_target_": "auementations.adapters.pedalboard.HighPassFilter",
                    "sample_rate": sample_rate,
                    "min_cutoff_freq": 100.0,
                    "max_cutoff_freq": 500.0,
                    "p": 1.0,
                },
            },
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
        results = [augmentation(mono_audio.clone()) for _ in range(10)]
        # Check that not all results are identical
        all_same = all(torch.allclose(results[0], r, atol=1e-6) for r in results[1:])
        assert not all_same, "All results identical - composition not working correctly"


class TestRealisticPeakFilterEQ:
    """Test realistic EQ usage with PeakFilter across different frequency bands."""

    def test_given_one_of_eq_bands_when_applied_then_randomly_selects_frequency_range(
        self,
    ):
        """Given OneOf with EQ bands, when applied, then randomly selects and applies filter from one band."""
        # GIVEN: A batch of audio with shape (batch, source, channel, time)
        batch_size = 4
        sources = 2
        channels = 1
        sample_rate = 48000
        duration = 1.0
        samples = int(sample_rate * duration)

        # Create test audio
        audio = torch.randn(batch_size, sources, channels, samples) * 0.5

        # Get the PeakFilter config
        peak_filter_config = auementations_store.get_entry(
            group="auementations/pedalboard", name="peak_filter"
        )["node"]
        one_of_config = auementations_store.get_entry(
            group="auementations/composition", name="one_of"
        )["node"]

        # Define realistic EQ bands used in audio production
        eq_bands = {
            "lows": {
                "min_center_freq": 60.0,
                "max_center_freq": 250.0,
                "min_gain_db": -6.0,
                "max_gain_db": 6.0,
                "min_q": 0.5,
                "max_q": 1.5,
            },
            "low_mids": {
                "min_center_freq": 250.0,
                "max_center_freq": 500.0,
                "min_gain_db": -6.0,
                "max_gain_db": 6.0,
                "min_q": 0.7,
                "max_q": 2.0,
            },
            "mids": {
                "min_center_freq": 500.0,
                "max_center_freq": 2000.0,
                "min_gain_db": -6.0,
                "max_gain_db": 6.0,
                "min_q": 0.7,
                "max_q": 2.0,
            },
            "upper_mids": {
                "min_center_freq": 2000.0,
                "max_center_freq": 6000.0,
                "min_gain_db": -6.0,
                "max_gain_db": 6.0,
                "min_q": 1.0,
                "max_q": 3.0,
            },
            "highs": {
                "min_center_freq": 6000.0,
                "max_center_freq": 20000.0,
                "min_gain_db": -6.0,
                "max_gain_db": 6.0,
                "min_q": 0.5,
                "max_q": 2.0,
            },
        }

        # WHEN: Build a OneOf composition with all EQ bands
        augmentations_dict = {}
        for band_name, band_params in eq_bands.items():
            augmentations_dict[band_name] = builds(
                peak_filter_config,
                sample_rate=sample_rate,
                mode="per_example",  # Different params per batch example
                p=1.0,
                **band_params,
            )

        composition_config = builds(
            one_of_config,
            augmentations=augmentations_dict,
            sample_rate=sample_rate,
            p=1.0,
        )

        augmentation = instantiate(composition_config)

        # THEN: The augmentation can be applied successfully
        result = augmentation(audio)
        assert result.shape == audio.shape
        assert isinstance(result, torch.Tensor)

        # AND: The result is different from input (filter was applied)
        assert not torch.allclose(result, audio, rtol=1e-3)

    def test_given_one_of_eq_bands_when_applied_multiple_times_then_produces_variation(
        self,
    ):
        """Given OneOf EQ composition, when applied multiple times, then produces different results."""
        # GIVEN: Audio and OneOf composition with multiple EQ bands
        batch_size = 2
        sources = 1
        channels = 1
        sample_rate = 48000
        duration = 0.5
        samples = int(sample_rate * duration)

        # Use a simple signal to better detect differences
        t = torch.linspace(0, duration, samples)
        # Combine multiple frequencies
        audio = (
            torch.sin(2 * torch.pi * 200 * t)
            + torch.sin(2 * torch.pi * 1000 * t)
            + torch.sin(2 * torch.pi * 4000 * t)
        )
        audio = (
            audio.view(1, 1, 1, -1).expand(batch_size, sources, channels, -1).clone()
        )
        audio = audio * 0.3

        peak_filter_config = auementations_store.get_entry(
            group="auementations/pedalboard", name="peak_filter"
        )["node"]
        one_of_config = auementations_store.get_entry(
            group="auementations/composition", name="one_of"
        )["node"]

        # Create composition with EQ bands
        composition_config = builds(
            one_of_config,
            augmentations={
                "lows": builds(
                    peak_filter_config,
                    sample_rate=sample_rate,
                    min_center_freq=60.0,
                    max_center_freq=250.0,
                    min_gain_db=-6.0,
                    max_gain_db=6.0,
                    min_q=0.5,
                    max_q=1.5,
                    mode="per_example",
                    p=1.0,
                ),
                "mids": builds(
                    peak_filter_config,
                    sample_rate=sample_rate,
                    min_center_freq=500.0,
                    max_center_freq=2000.0,
                    min_gain_db=-6.0,
                    max_gain_db=6.0,
                    min_q=0.7,
                    max_q=2.0,
                    mode="per_example",
                    p=1.0,
                ),
                "highs": builds(
                    peak_filter_config,
                    sample_rate=sample_rate,
                    min_center_freq=6000.0,
                    max_center_freq=20000.0,
                    min_gain_db=-6.0,
                    max_gain_db=6.0,
                    min_q=0.5,
                    max_q=2.0,
                    mode="per_example",
                    p=1.0,
                ),
            },
            sample_rate=sample_rate,
            p=1.0,
            seed=None,  # Allow true randomness
        )

        augmentation = instantiate(composition_config)

        # WHEN: Apply multiple times
        num_trials = 15
        results = []
        for _ in range(num_trials):
            result = augmentation(audio.clone())
            results.append(result)

        # THEN: Should produce variation across trials
        # Count unique results (comparing first batch element)
        unique_results = []
        for result in results:
            is_unique = True
            result_sample = result[0, 0, 0, :100]  # Compare first 100 samples
            for unique in unique_results:
                if torch.allclose(result_sample, unique, atol=1e-4):
                    is_unique = False
                    break
            if is_unique:
                unique_results.append(result_sample)

        # Should have multiple unique results due to:
        # 1. Random selection of EQ band
        # 2. Random frequency within band
        # 3. Random gain
        # 4. Random Q
        assert len(unique_results) > 3, (
            f"Expected variation from randomness, but only got {len(unique_results)} unique results"
        )

    def test_given_per_example_mode_when_applied_to_batch_then_each_example_gets_different_eq(
        self,
    ):
        """Given per_example mode, when applied to batch, then each example receives different EQ."""
        # GIVEN: A batch with identical audio in each example
        batch_size = 4
        sources = 1
        channels = 1
        sample_rate = 48000
        duration = 0.5
        samples = int(sample_rate * duration)

        # Use identical signal for all examples to highlight differences
        t = torch.linspace(0, duration, samples)
        signal = torch.sin(2 * torch.pi * 1000 * t) * 0.5
        audio = (
            signal.view(1, 1, 1, -1).expand(batch_size, sources, channels, -1).clone()
        )

        peak_filter_config = auementations_store.get_entry(
            group="auementations/pedalboard", name="peak_filter"
        )["node"]

        # Create a PeakFilter that targets mids with per_example mode
        augmentation = instantiate(
            builds(
                peak_filter_config,
                sample_rate=sample_rate,
                min_center_freq=500.0,
                max_center_freq=2000.0,
                min_gain_db=-6.0,
                max_gain_db=6.0,
                min_q=0.7,
                max_q=2.0,
                mode="per_example",
                p=1.0,
                seed=42,
            )
        )

        # WHEN: Apply to the batch
        result = augmentation(audio)

        # THEN: Each example should be different due to per_example mode
        outputs_per_example = []
        for i in range(batch_size):
            outputs_per_example.append(result[i, 0, 0, :100].detach().numpy())

        # Check that examples are different from each other
        differences_found = 0
        for i in range(batch_size - 1):
            for j in range(i + 1, batch_size):
                if not np.allclose(
                    outputs_per_example[i], outputs_per_example[j], rtol=1e-3
                ):
                    differences_found += 1

        # With 4 examples, we have 6 pairwise comparisons
        # We should see most or all of them being different
        assert differences_found >= 4, (
            f"Expected different EQ per example, but only found {differences_found} differences"
        )


@pytest.mark.parametrize("aug_name", ["lpf", "hpf"])
@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_PedalboardAugmentationsHandle2_3_4_dims(aug_name, stereo_audio, ndim):
    sr = 16000
    config = auementations_store.get_entry(
        group="auementations/pedalboard", name=aug_name
    )["node"]

    # composition_config = builds(
    #     config,
    # )
    #
    # stereo_audio is (1, 2, time) = (source, channel, time) - 3D
    match ndim:
        case 2:
            # For 2D, extract just (channel, time)
            input = stereo_audio[0]  # (2, time)
        case 3:
            # For 3D, use as-is: (source, channel, time)
            input = stereo_audio
        case 4:
            # For 4D, add batch dimension: (batch, source, channel, time)
            input = stereo_audio.unsqueeze(0)
        case _:
            input = stereo_audio

    augmentation = instantiate(config, sample_rate=sr, p=1.0)

    y_hat = augmentation(input)
    assert input.shape == y_hat.shape
