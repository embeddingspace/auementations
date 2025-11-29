"""BDD-style tests for centralized hydra-zen store functionality."""

import pytest
from hydra_zen import ZenStore, instantiate


class TestAuemStoreCreation:
    """Test scenarios for AUEM store creation and basic structure."""

    def test_given_auem_package_when_importing_store_then_store_exists(self):
        """Given the auem package, when importing the store, then it exists and is a ZenStore."""
        # Given / When
        from auementations.config import auementations_store

        # Then
        assert auementations_store is not None
        assert isinstance(auementations_store, ZenStore)

    def test_given_auem_store_when_checking_name_then_has_auementations_name(self):
        """Given the Auementations store, when checking its name, then it is named 'auementations'."""
        # Given
        from auementations.config import auementations_store

        # When
        store_name = auementations_store.name

        # Then
        assert store_name == "auementations"

    def test_given_auem_store_when_accessing_then_is_singleton(self):
        """Given the Auementations store, when importing multiple times, then returns same instance."""
        # Given / When
        from auementations.config import auementations_store as store1
        from auementations.config import auementations_store as store2

        # Then
        assert store1 is store2


class TestStoreCompositionConfigs:
    """Test scenarios for composition config registration in the store."""

    def test_given_store_when_checking_compose_config_then_is_registered(self):
        """Given the store, when checking for Compose config, then it is registered."""
        # Given
        from auementations.config import auementations_store

        # When
        entry = auementations_store.get_entry("auementations/composition", "compose")

        # Then
        assert entry is not None
        assert entry["name"] == "compose"
        assert entry["group"] == "auementations/composition"

    def test_given_store_when_checking_oneof_config_then_is_registered(self):
        """Given the store, when checking for OneOf config, then it is registered."""
        # Given
        from auementations.config import auementations_store

        # When
        entry = auementations_store.get_entry("auementations/composition", "one_of")

        # Then
        assert entry is not None
        assert entry["name"] == "one_of"

    def test_given_store_when_checking_someof_config_then_is_registered(self):
        """Given the store, when checking for SomeOf config, then it is registered."""
        # Given
        from auementations.config import auementations_store

        # When
        entry = auementations_store.get_entry("auementations/composition", "some_of")

        # Then
        assert entry is not None
        assert entry["name"] == "some_of"

    def test_given_compose_config_when_instantiating_then_creates_compose_instance(
        self, sample_rate
    ):
        """Given Compose config from store, when instantiating, then creates Compose object."""
        # Given
        from auementations.config import auementations_store
        from auementations.core.composition import Compose
        from tests.conftest import MockAugmentation

        compose_config = auementations_store(
            group="auementations/composition", name="compose"
        )

        # Create a config with actual augmentations
        config_dict = {
            "_target_": "auementations.core.composition.Compose",
            "augmentations": {
                "mock": MockAugmentation(sample_rate=sample_rate, gain=2.0),
            },
            "sample_rate": sample_rate,
            "p": 1.0,
        }

        # When
        instance = instantiate(config_dict)

        # Then
        assert isinstance(instance, Compose)
        assert instance.sample_rate == sample_rate


class TestStoreTorchAudiomentationsConfigs:
    """Test scenarios for torch_audiomentations adapter config registration."""

    def test_given_store_when_checking_gain_config_then_is_registered(self):
        """Given the store, when checking for Gain config, then it is registered."""
        # Given
        from auementations.config import auementations_store

        # When
        entry = auementations_store.get_entry(
            "auementations/torch_audiomentations", "gain"
        )

        # Then
        assert entry is not None
        assert entry["name"] == "gain"
        assert entry["group"] == "auementations/torch_audiomentations"

    def test_given_store_when_checking_pitch_shift_config_then_is_registered(self):
        """Given the store, when checking for PitchShift config, then it is registered."""
        # Given
        from auementations.config import auementations_store

        # When
        entry = auementations_store.get_entry(
            "auementations/torch_audiomentations", "pitch_shift"
        )

        # Then
        assert entry is not None
        assert entry["name"] == "pitch_shift"

    def test_given_store_when_checking_add_colored_noise_config_then_is_registered(
        self,
    ):
        """Given the store, when checking for AddColoredNoise config, then it is registered."""
        # Given
        from auementations.config import auementations_store

        # When
        entry = auementations_store.get_entry(
            "auementations/torch_audiomentations", "add_colored_noise"
        )

        # Then
        assert entry is not None

    def test_given_store_when_checking_high_pass_filter_config_then_is_registered(self):
        """Given the store, when checking for HighPassFilter config, then it is registered."""
        # Given
        from auementations.config import auementations_store

        # When
        entry = auementations_store.get_entry(
            "auementations/torch_audiomentations", "hpf"
        )

        # Then
        assert entry is not None

    def test_given_store_when_checking_low_pass_filter_config_then_is_registered(self):
        """Given the store, when checking for LowPassFilter config, then it is registered."""
        # Given
        from auementations.config import auementations_store

        # When
        entry = auementations_store.get_entry(
            "auementations/torch_audiomentations", "lpf"
        )

        # Then
        assert entry is not None

    def test_given_store_when_checking_time_stretch_config_then_is_registered(self):
        """Given the store, when checking for TimeStretch config, then it is registered."""
        # Given
        from auementations.config import auementations_store

        # When
        entry = auementations_store.get_entry(
            "auementations/torch_audiomentations", "time_stretch"
        )

        # Then
        assert entry is not None


class TestStoreExternalRepoUsage:
    """Test scenarios for external repositories using the AUEM store."""

    def test_given_auem_store_when_merging_to_external_store_then_configs_available(
        self,
    ):
        """Given Auementations store, when merging to external store, then configs are available."""
        # Given
        from auementations.config import auementations_store

        external_store = ZenStore(name="external_project")

        # When
        merged_store = external_store.merge(auementations_store)

        # Then
        # Merged store should have Auementations configs
        assert (
            merged_store.get_entry("auementations/composition", "compose") is not None
        )
        assert merged_store.get_entry("auementations/composition", "one_of") is not None
        assert (
            merged_store.get_entry("auementations/composition", "some_of") is not None
        )

    def test_given_merged_store_when_adding_custom_configs_then_both_available(self):
        """Given merged store, when adding custom configs, then both Auementations and custom configs available."""
        # Given
        from hydra_zen import builds

        from auementations.config import auementations_store
        from tests.conftest import MockAugmentation

        external_store = ZenStore(name="external_project")
        merged_store = external_store.merge(auementations_store)

        # When
        # Add a custom config to the merged store
        custom_config = builds(
            MockAugmentation,
            sample_rate=16000,
            gain=5.0,
            populate_full_signature=True,
        )
        merged_store(custom_config, group="custom", name="custom_mock")

        # Then
        # Both Auementations and custom configs should be available
        assert (
            merged_store.get_entry("auementations/composition", "compose") is not None
        )
        assert merged_store.get_entry("custom", "custom_mock") is not None
