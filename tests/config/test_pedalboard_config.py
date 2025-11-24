"""Tests for pedalboard configuration registration in ZenStore."""

import pytest

pytest.importorskip("pedalboard")

from hydra_zen import instantiate

from auementations.config.config_store import auementations_store


class TestPedalboardConfigRegistration:
    """Test scenarios for pedalboard config registration in ZenStore."""

    def test_given_lpf_config_when_retrieved_from_store_then_found(self):
        """Given LPF config, when retrieved from store, then it is found."""
        # When
        configs = auementations_store.get_entry(
            group="augmentation/pedalboard", name="lpf"
        )

        # Then
        assert configs is not None
        assert "node" in configs

    def test_given_hpf_config_when_retrieved_from_store_then_found(self):
        """Given HPF config, when retrieved from store, then it is found."""
        # When
        configs = auementations_store.get_entry(
            group="augmentation/pedalboard", name="hpf"
        )

        # Then
        assert configs is not None
        assert "node" in configs

    def test_given_lpf_config_when_instantiated_then_creates_lowpass_filter(self):
        """Given LPF config, when instantiated, then creates LowPassFilter instance."""
        # Given
        configs = auementations_store.get_entry(
            group="augmentation/pedalboard", name="lpf"
        )

        # When
        lpf = instantiate(configs["node"], sample_rate=16000, cutoff_freq=1000.0)

        # Then
        from auementations.adapters.pedalboard import LowPassFilter

        assert isinstance(lpf, LowPassFilter)
        assert lpf.sample_rate == 16000

    def test_given_hpf_config_when_instantiated_then_creates_highpass_filter(self):
        """Given HPF config, when instantiated, then creates HighPassFilter instance."""
        # Given
        configs = auementations_store.get_entry(
            group="augmentation/pedalboard", name="hpf"
        )

        # When
        hpf = instantiate(configs["node"], sample_rate=16000, cutoff_freq=500.0)

        # Then
        from auementations.adapters.pedalboard import HighPassFilter

        assert isinstance(hpf, HighPassFilter)
        assert hpf.sample_rate == 16000

    def test_given_lpf_config_when_instantiated_with_range_then_handles_range_parameters(
        self,
    ):
        """Given LPF config, when instantiated with range, then handles range parameters correctly."""
        # Given
        configs = auementations_store.get_entry(
            group="augmentation/pedalboard", name="lpf"
        )

        # When
        lpf = instantiate(
            configs["node"], sample_rate=44100, min_cutoff_freq=500.0, max_cutoff_freq=5000.0
        )

        # Then
        from auementations.adapters.pedalboard import LowPassFilter

        assert isinstance(lpf, LowPassFilter)
        assert lpf.sample_rate == 44100
        assert lpf.param_specs["cutoff_frequency_hz"] == (500.0, 5000.0)
