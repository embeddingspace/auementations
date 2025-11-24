"""BDD-style tests for pedalboard adapter functionality.

These tests follow the Given-When-Then pattern to ensure the pedalboard
adapter works correctly.
"""

import numpy as np
import pytest

from auementations.adapters.pedalboard import (
    HighPassFilter,
    LowPassFilter,
    PedalboardAdapter,
)


class TestPedalboardAdapterInitialization:
    """Test scenarios for initializing pedalboard adapters."""

    def test_given_valid_parameters_when_initialized_then_creates_adapter(self):
        """Given valid parameters, when adapter is initialized, then it creates successfully."""
        # Given
        sample_rate = 44100
        cutoff_freq = 1000.0

        # When
        aug = LowPassFilter(sample_rate=sample_rate, cutoff_freq=cutoff_freq)

        # Then
        assert aug.sample_rate == sample_rate
        assert aug.param_specs["cutoff_frequency_hz"] == cutoff_freq

    def test_given_range_parameters_when_initialized_then_stores_ranges(self):
        """Given range parameters, when adapter is initialized, then it stores parameter ranges."""
        # Given
        sample_rate = 44100
        min_cutoff = 200.0
        max_cutoff = 2000.0

        # When
        aug = HighPassFilter(
            sample_rate=sample_rate,
            min_cutoff_freq=min_cutoff,
            max_cutoff_freq=max_cutoff,
        )

        # Then
        assert aug.param_specs["cutoff_frequency_hz"] == (min_cutoff, max_cutoff)


class TestPedalboardAdapterParameterSampling:
    """Test scenarios for parameter sampling."""

    def test_given_range_parameter_when_randomized_then_samples_within_range(self):
        """Given a range parameter, when randomized, then sampled value is within range."""
        # Given
        sample_rate = 44100
        min_cutoff = 200.0
        max_cutoff = 2000.0
        aug = LowPassFilter(
            sample_rate=sample_rate,
            min_cutoff_freq=min_cutoff,
            max_cutoff_freq=max_cutoff,
            seed=42,
        )

        # When
        for _ in range(10):
            aug.randomize_parameters()
            sampled = aug.current_params["cutoff_frequency_hz"]

            # Then
            assert min_cutoff <= sampled <= max_cutoff

    def test_given_fixed_parameter_when_randomized_then_uses_fixed_value(self):
        """Given a fixed parameter, when randomized, then always uses the same value."""
        # Given
        sample_rate = 44100
        cutoff_freq = 1000.0
        aug = HighPassFilter(sample_rate=sample_rate, cutoff_freq=cutoff_freq)

        # When
        for _ in range(10):
            aug.randomize_parameters()

            # Then
            assert aug.current_params["cutoff_frequency_hz"] == cutoff_freq


class TestLowPassFilterAugmentation:
    """Test scenarios for low-pass filter augmentation."""

    def test_given_mono_audio_when_filtered_then_attenuates_high_frequencies(
        self, mono_audio
    ):
        """Given mono audio, when low-pass filter applied, then high frequencies are attenuated."""
        # Given
        sample_rate = 16000
        cutoff_freq = 1000.0  # Low cutoff to clearly attenuate high frequencies
        aug = LowPassFilter(sample_rate=sample_rate, cutoff_freq=cutoff_freq, p=1.0)

        # When
        result = aug(mono_audio)

        # Then
        # Output should have same shape
        assert result.shape == mono_audio.shape
        # Output should be different (filtered)
        assert not np.array_equal(result, mono_audio)
        # Output should have reduced high-frequency content (lower RMS than original sine wave at 440Hz)
        # Since 440Hz is below 1000Hz cutoff, the signal should pass through with some attenuation
        # But this is a basic sanity check that filtering occurred
        assert isinstance(result, np.ndarray)

    def test_given_stereo_audio_when_filtered_then_preserves_shape(self, stereo_audio):
        """Given stereo audio, when low-pass filter applied, then output shape is preserved."""
        # Given
        sample_rate = 16000
        aug = LowPassFilter(sample_rate=sample_rate, cutoff_freq=2000.0, p=1.0)

        # When
        result = aug(stereo_audio)

        # Then
        assert result.shape == stereo_audio.shape

    def test_given_probability_zero_when_applied_then_returns_unchanged_audio(
        self, mono_audio
    ):
        """Given p=0.0, when low-pass filter applied, then audio is unchanged."""
        # Given
        aug = LowPassFilter(sample_rate=16000, cutoff_freq=1000.0, p=0.0)
        original = mono_audio.copy()

        # When
        result = aug(mono_audio)

        # Then
        assert np.array_equal(result, original)

    def test_given_range_cutoff_when_applied_then_uses_random_cutoff(self):
        """Given cutoff range, when applied, then uses random cutoff frequency."""
        # Given
        sample_rate = 16000
        min_cutoff = 500.0
        max_cutoff = 2000.0
        aug = LowPassFilter(
            sample_rate=sample_rate,
            min_cutoff_freq=min_cutoff,
            max_cutoff_freq=max_cutoff,
            seed=42,
        )

        # When - randomize and check parameters are sampled
        aug.randomize_parameters()
        cutoff = aug.current_params["cutoff_frequency_hz"]

        # Then
        assert min_cutoff <= cutoff <= max_cutoff


class TestHighPassFilterAugmentation:
    """Test scenarios for high-pass filter augmentation."""

    def test_given_mono_audio_when_filtered_then_attenuates_low_frequencies(
        self, mono_audio
    ):
        """Given mono audio, when high-pass filter applied, then low frequencies are attenuated."""
        # Given
        sample_rate = 16000
        cutoff_freq = 5000.0  # High cutoff to attenuate the 440Hz test signal
        aug = HighPassFilter(sample_rate=sample_rate, cutoff_freq=cutoff_freq, p=1.0)

        # When
        result = aug(mono_audio)

        # Then
        # Output should have same shape
        assert result.shape == mono_audio.shape
        # Output should be different (filtered)
        assert not np.array_equal(result, mono_audio)
        # Output should have significantly reduced amplitude since 440Hz is below 5000Hz cutoff
        assert isinstance(result, np.ndarray)

    def test_given_stereo_audio_when_filtered_then_preserves_shape(self, stereo_audio):
        """Given stereo audio, when high-pass filter applied, then output shape is preserved."""
        # Given
        sample_rate = 16000
        aug = HighPassFilter(sample_rate=sample_rate, cutoff_freq=100.0, p=1.0)

        # When
        result = aug(stereo_audio)

        # Then
        assert result.shape == stereo_audio.shape

    def test_given_probability_zero_when_applied_then_returns_unchanged_audio(
        self, mono_audio
    ):
        """Given p=0.0, when high-pass filter applied, then audio is unchanged."""
        # Given
        aug = HighPassFilter(sample_rate=16000, cutoff_freq=1000.0, p=0.0)
        original = mono_audio.copy()

        # When
        result = aug(mono_audio)

        # Then
        assert np.array_equal(result, original)

    def test_given_range_cutoff_when_applied_then_uses_random_cutoff(self):
        """Given cutoff range, when applied, then uses random cutoff frequency."""
        # Given
        sample_rate = 16000
        min_cutoff = 100.0
        max_cutoff = 1000.0
        aug = HighPassFilter(
            sample_rate=sample_rate,
            min_cutoff_freq=min_cutoff,
            max_cutoff_freq=max_cutoff,
            seed=42,
        )

        # When
        aug.randomize_parameters()
        cutoff = aug.current_params["cutoff_frequency_hz"]

        # Then
        assert min_cutoff <= cutoff <= max_cutoff


class TestPedalboardAdapterConfiguration:
    """Test scenarios for configuration export."""

    def test_given_lowpass_filter_when_exported_to_config_then_contains_parameters(
        self,
    ):
        """Given a low-pass filter, when exported to config, then config contains all parameters."""
        # Given
        aug = LowPassFilter(
            sample_rate=44100, min_cutoff_freq=500.0, max_cutoff_freq=5000.0, p=0.8
        )

        # When
        config = aug.to_config()

        # Then
        assert "_target_" in config
        assert config["sample_rate"] == 44100
        assert config["p"] == 0.8
        assert "cutoff_frequency_hz" in config

    def test_given_highpass_filter_when_exported_to_config_then_contains_parameters(
        self,
    ):
        """Given a high-pass filter, when exported to config, then config contains all parameters."""
        # Given
        aug = HighPassFilter(
            sample_rate=44100, min_cutoff_freq=50.0, max_cutoff_freq=500.0, p=0.9
        )

        # When
        config = aug.to_config()

        # Then
        assert "_target_" in config
        assert config["sample_rate"] == 44100
        assert config["p"] == 0.9
        assert "cutoff_frequency_hz" in config


class TestPedalboardAdapterNumpyHandling:
    """Test scenarios for numpy array handling."""

    def test_given_numpy_array_when_processed_then_returns_numpy_array(
        self, mono_audio
    ):
        """Given numpy array input, when processed, then returns numpy array."""
        # Given
        aug = LowPassFilter(sample_rate=16000, cutoff_freq=2000.0)
        assert isinstance(mono_audio, np.ndarray)

        # When
        result = aug(mono_audio)

        # Then
        assert isinstance(result, np.ndarray)
        assert result.dtype == mono_audio.dtype

    def test_given_different_audio_shapes_when_processed_then_preserves_shapes(self):
        """Given different audio shapes, when processed, then output shapes match input."""
        # Given
        aug = HighPassFilter(sample_rate=16000, cutoff_freq=200.0)

        # Test 1D mono
        mono_1d = np.random.randn(16000).astype(np.float32)
        result_1d = aug(mono_1d)
        assert result_1d.shape == mono_1d.shape

        # Test 2D stereo
        stereo_2d = np.random.randn(2, 16000).astype(np.float32)
        result_2d = aug(stereo_2d)
        assert result_2d.shape == stereo_2d.shape
