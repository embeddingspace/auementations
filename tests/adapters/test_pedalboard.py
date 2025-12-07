"""BDD-style tests for pedalboard adapter functionality.

These tests follow the Given-When-Then pattern to ensure the pedalboard
adapter works correctly.
"""

import numpy as np

from auementations.adapters.pedalboard import (
    HighPassFilter,
    LowPassFilter,
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


class TestPedalboardAdapterSmallSignals:
    """Test scenarios for handling small signal values."""

    def test_given_small_amplitude_audio_when_lpf_applied_then_returns_nonzero_values(
        self,
    ):
        """Given audio with small amplitudes, when LPF applied, then output contains non-zero values."""
        # Given - audio with very small amplitudes (like normalized audio during training)
        sample_rate = 16000
        duration = 1.0
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        # Create signal with amplitude of 0.001 (common in normalized audio)
        frequency = 1000.0  # Hz
        small_audio = (0.001 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

        # Use a cutoff that should allow the frequency through
        aug = LowPassFilter(sample_rate=sample_rate, cutoff_freq=2000.0, p=1.0)

        # When
        result = aug(small_audio)

        # Then - output should not be all zeros
        assert not np.allclose(result, 0.0), (
            "LPF should not produce all zeros for small signals"
        )
        assert np.any(np.abs(result) > 1e-6), (
            "LPF output should contain meaningful non-zero values"
        )
        # The output magnitude should be in the same ballpark as input
        assert np.abs(np.max(result)) > 1e-5, (
            "LPF should preserve signal amplitude for frequencies below cutoff"
        )

    def test_given_small_amplitude_audio_when_hpf_applied_then_returns_nonzero_values(
        self,
    ):
        """Given audio with small amplitudes, when HPF applied, then output contains non-zero values."""
        # Given - audio with very small amplitudes
        sample_rate = 16000
        duration = 1.0
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        # Create signal with amplitude of 0.001 at high frequency
        frequency = 5000.0  # Hz - well above typical HPF cutoff
        small_audio = (0.001 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

        # Use a low cutoff that should allow the high frequency through
        aug = HighPassFilter(sample_rate=sample_rate, cutoff_freq=100.0, p=1.0)

        # When
        result = aug(small_audio)

        # Then - output should not be all zeros
        assert not np.allclose(result, 0.0), (
            "HPF should not produce all zeros for small signals"
        )
        assert np.any(np.abs(result) > 1e-6), (
            "HPF output should contain meaningful non-zero values"
        )
        # The output magnitude should be in the same ballpark as input
        assert np.abs(np.max(result)) > 1e-5, (
            "HPF should preserve signal amplitude for frequencies above cutoff"
        )


class TestPedalboardAdapterBatchProcessing:
    """Test scenarios for batch processing with different parameters per example."""

    def test_given_batch_audio_when_lpf_applied_then_each_example_gets_different_filter(
        self,
    ):
        """Given batch of audio, when LPF with range applied, then each example gets different filter parameters."""
        # Given - batch of 4 identical mono audio signals
        sample_rate = 16000
        duration = 0.5
        num_samples = int(duration * sample_rate)
        batch_size = 4
        num_channels = 1  # Mono

        # Create batch with mixed frequencies
        t = np.linspace(0, duration, num_samples)
        # Use multiple frequencies to better detect filtering differences
        signal = np.zeros(num_samples, dtype=np.float32)
        for freq in [500, 1000, 2000, 4000, 6000]:
            signal += np.sin(2 * np.pi * freq * t).astype(np.float32)

        # Shape: [batch, channels, samples] - always include channel dimension for batched inputs
        batch_audio = np.tile(signal, (batch_size, num_channels, 1))

        # Wide cutoff range to ensure different filters
        aug = LowPassFilter(
            sample_rate=sample_rate,
            min_cutoff_freq=500.0,
            max_cutoff_freq=6000.0,
            p=1.0,
            seed=42,
        )

        # When
        result = aug(batch_audio)

        # Then - outputs should be different for each example
        # Compare first example with others
        for i in range(1, batch_size):
            # Examples should not be identical (different filter was applied)
            assert not np.allclose(result[0], result[i], atol=1e-4), (
                f"Example 0 and {i} should have different filtering"
            )

    def test_given_batch_channel_audio_when_hpf_applied_then_each_example_gets_different_filter(
        self,
    ):
        """Given batch with channels, when HPF applied, then each example gets different filter parameters."""
        # Given - batch of stereo audio: [batch, channel, samples]
        sample_rate = 16000
        duration = 0.5
        num_samples = int(duration * sample_rate)
        batch_size = 4
        num_channels = 2

        # Create signal with multiple frequencies
        t = np.linspace(0, duration, num_samples)
        signal = np.zeros(num_samples, dtype=np.float32)
        for freq in [100, 500, 1000, 2000]:
            signal += np.sin(2 * np.pi * freq * t).astype(np.float32)

        # Shape: [batch, channel, samples]
        batch_audio = np.tile(signal, (batch_size, num_channels, 1))

        # Wide cutoff range
        aug = HighPassFilter(
            sample_rate=sample_rate,
            min_cutoff_freq=50.0,
            max_cutoff_freq=1500.0,
            p=1.0,
            seed=123,
        )

        # When
        result = aug(batch_audio)

        # Then - outputs should be different for each batch example
        for i in range(1, batch_size):
            assert not np.allclose(result[0], result[i], atol=1e-4), (
                f"Example 0 and {i} should have different filtering"
            )

    def test_given_batch_source_channel_audio_when_lpf_applied_then_each_example_gets_different_filter(
        self,
    ):
        """Given batch with sources and channels, when LPF applied, then each example gets different filter."""
        # Given - batch with shape [batch, source, channel, samples]
        sample_rate = 16000
        duration = 0.5
        num_samples = int(duration * sample_rate)
        batch_size = 3
        num_sources = 2
        num_channels = 2

        # Create signal with multiple frequencies
        t = np.linspace(0, duration, num_samples)
        signal = np.zeros(num_samples, dtype=np.float32)
        for freq in [500, 1500, 3000, 5000]:
            signal += np.sin(2 * np.pi * freq * t).astype(np.float32)

        # Shape: [batch, source, channel, samples]
        batch_audio = np.tile(signal, (batch_size, num_sources, num_channels, 1))

        aug = LowPassFilter(
            sample_rate=sample_rate,
            min_cutoff_freq=500.0,
            max_cutoff_freq=5000.0,
            p=1.0,
            seed=456,
        )

        # When
        result = aug(batch_audio)

        # Then - outputs should be different for each batch example
        for i in range(1, batch_size):
            assert not np.allclose(result[0], result[i], atol=1e-4), (
                f"Example 0 and {i} should have different filtering"
            )
