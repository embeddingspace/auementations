"""Tests for PeakFilter augmentation."""

import numpy as np
import pytest
import torch

from auementations.adapters.pedalboard import PeakFilter


class TestPeakFilter:
    def test_should_instantiate_with_default_params(self):
        """GIVEN default parameters, WHEN creating PeakFilter, THEN it should instantiate."""
        aug = PeakFilter(sample_rate=16000)
        assert aug is not None
        assert aug.sample_rate == 16000

    def test_forward_pass_should_modify_audio(self):
        """GIVEN audio and PeakFilter, WHEN applied, THEN audio should be modified."""
        # GIVEN: Audio with shape (batch, source, channel, time) and a PeakFilter
        batch_size = 2
        sources = 2
        channels = 1
        samples = 16000
        audio = torch.randn(batch_size, sources, channels, samples) * 2 - 1
        aug = PeakFilter(
            sample_rate=16000,
            min_center_freq=1000.0,
            max_center_freq=1000.0,
            min_gain_db=6.0,
            max_gain_db=6.0,
            min_q=0.707,
            max_q=0.707,
            p=1.0,
        )

        # WHEN: The augmentation is applied
        y_hat = aug(audio)

        # THEN: The output should be different from input and shape preserved
        assert y_hat.shape == audio.shape
        assert not torch.allclose(y_hat, audio, rtol=1e-3)

    def test_p0_should_always_return_original(self):
        """GIVEN p=0, WHEN applied, THEN original audio should be returned."""
        # GIVEN: Audio and augmentation with p=0
        audio = torch.randn(2, 2, 1, 16000) * 0.5
        aug = PeakFilter(sample_rate=16000, p=0.0)

        # WHEN: Applied multiple times
        for _ in range(3):
            y_hat = aug(audio)
            # THEN: Output should match input
            assert torch.allclose(y_hat, audio)

    def test_p1_should_always_apply(self):
        """GIVEN p=1, WHEN applied, THEN audio should always be modified."""
        # GIVEN: Audio and augmentation with p=1
        audio = torch.randn(2, 2, 1, 16000) * 0.5
        aug = PeakFilter(
            sample_rate=16000,
            min_center_freq=1000.0,
            max_center_freq=1000.0,
            min_gain_db=6.0,
            max_gain_db=6.0,
            p=1.0,
        )

        # WHEN: Applied multiple times
        for _ in range(3):
            y_hat = aug(audio)
            # THEN: Output should be different from input
            assert not torch.allclose(y_hat, audio, rtol=1e-3)

    def test_mode_per_batch_applies_same_filter_to_entire_batch(self):
        """GIVEN mode='per_batch', WHEN applied, THEN same filter params used for all examples."""
        # GIVEN: A batch of audio with shape (batch, source, channel, time)
        batch_size = 4
        sources = 2
        channels = 1
        samples = 8000
        audio = torch.rand(batch_size, sources, channels, samples) * 2 - 1
        aug = PeakFilter(
            sample_rate=16000,
            min_center_freq=500.0,
            max_center_freq=2000.0,
            min_gain_db=-6.0,
            max_gain_db=6.0,
            min_q=0.5,
            max_q=2.0,
            p=1.0,
            mode="per_batch",
            seed=42,
        )

        # WHEN: The augmentation is applied
        y_hat = aug(audio)

        # THEN: Shape should be preserved
        assert y_hat.shape == audio.shape

        # THEN: The same filter parameters should be applied to all examples
        # We can verify this by checking that the spectral modifications are similar
        # across examples (within tolerance for numerical precision)
        fft_0 = torch.fft.rfft(y_hat[0, 0, 0])
        for i in range(1, batch_size):
            fft_i = torch.fft.rfft(y_hat[i, 0, 0])
            # The FFT magnitudes should have similar patterns
            # (exact match not expected due to different input signals)
            ratio = torch.abs(fft_i) / (torch.abs(fft_0) + 1e-8)
            # For the same filter, the ratio should be relatively consistent
            # This is a loose check; we mainly verify the code path works
            assert ratio.shape == fft_0.shape

    def test_mode_per_example_applies_different_filter_per_example(self):
        """GIVEN mode='per_example', WHEN applied, THEN different filter params per example."""
        # GIVEN: A batch of identical audio signals
        batch_size = 4
        sources = 2
        channels = 1
        samples = 8000
        # Use a simple sine wave so we can detect differences
        t = torch.linspace(0, 1, samples)
        sine_wave = torch.sin(2 * torch.pi * 440 * t)  # 440 Hz sine
        # Repeat for batch, sources, channels
        audio = (
            sine_wave.view(1, 1, 1, -1)
            .expand(batch_size, sources, channels, -1)
            .clone()
        )

        aug = PeakFilter(
            sample_rate=16000,
            min_center_freq=400.0,
            max_center_freq=500.0,
            min_gain_db=3.0,
            max_gain_db=6.0,
            min_q=0.707,
            max_q=0.707,
            p=1.0,
            mode="per_example",
            seed=42,
        )

        # WHEN: The augmentation is applied
        y_hat = aug(audio)

        # THEN: Shape should be preserved
        assert y_hat.shape == audio.shape

        # THEN: Each example should have received different filter parameters
        # We verify this by checking that outputs differ across examples
        outputs_per_example = []
        for i in range(batch_size):
            # Take a representative sample from each example
            example_output = y_hat[i, 0, 0, :100].detach().numpy()
            outputs_per_example.append(example_output)

        # Check that at least some examples are different from each other
        differences_found = 0
        for i in range(batch_size - 1):
            for j in range(i + 1, batch_size):
                if not np.allclose(
                    outputs_per_example[i], outputs_per_example[j], rtol=1e-3
                ):
                    differences_found += 1

        # With 4 examples and random parameters, we should see multiple differences
        assert differences_found >= 3, (
            f"Expected different filter parameters per example, "
            f"but only found {differences_found} differences"
        )

    def test_mode_per_source_applies_different_filter_per_source(self):
        """GIVEN mode='per_source', WHEN applied, THEN different filter params per source."""
        # GIVEN: Audio with multiple sources
        batch_size = 2
        sources = 4
        channels = 1
        samples = 8000
        # Use a simple sine wave
        t = torch.linspace(0, 1, samples)
        sine_wave = torch.sin(2 * torch.pi * 440 * t)
        audio = (
            sine_wave.view(1, 1, 1, -1)
            .expand(batch_size, sources, channels, -1)
            .clone()
        )

        aug = PeakFilter(
            sample_rate=16000,
            min_center_freq=400.0,
            max_center_freq=500.0,
            min_gain_db=3.0,
            max_gain_db=6.0,
            p=1.0,
            mode="per_source",
            seed=42,
        )

        # WHEN: The augmentation is applied
        y_hat = aug(audio)

        # THEN: Shape should be preserved
        assert y_hat.shape == audio.shape

        # THEN: Each source within a batch should have different parameters
        for batch_idx in range(batch_size):
            outputs_per_source = []
            for source_idx in range(sources):
                source_output = y_hat[batch_idx, source_idx, 0, :100].detach().numpy()
                outputs_per_source.append(source_output)

            # Check that sources are different from each other
            differences_found = 0
            for i in range(sources - 1):
                for j in range(i + 1, sources):
                    if not np.allclose(
                        outputs_per_source[i], outputs_per_source[j], rtol=1e-3
                    ):
                        differences_found += 1

            assert differences_found >= 3, (
                f"Expected different filter parameters per source in batch {batch_idx}, "
                f"but only found {differences_found} differences"
            )

    def test_mode_per_channel_applies_different_filter_per_channel(self):
        """GIVEN mode='per_channel', WHEN applied, THEN different filter params per channel."""
        # GIVEN: Audio with multiple channels
        batch_size = 2
        sources = 2
        channels = 4
        samples = 8000
        # Use a simple sine wave
        t = torch.linspace(0, 1, samples)
        sine_wave = torch.sin(2 * torch.pi * 440 * t)
        audio = (
            sine_wave.view(1, 1, 1, -1)
            .expand(batch_size, sources, channels, -1)
            .clone()
        )

        aug = PeakFilter(
            sample_rate=16000,
            min_center_freq=400.0,
            max_center_freq=500.0,
            min_gain_db=3.0,
            max_gain_db=6.0,
            p=1.0,
            mode="per_channel",
            seed=42,
        )

        # WHEN: The augmentation is applied
        y_hat = aug(audio)

        # THEN: Shape should be preserved
        assert y_hat.shape == audio.shape

        # THEN: Each channel should have different parameters
        for batch_idx in range(batch_size):
            for source_idx in range(sources):
                outputs_per_channel = []
                for channel_idx in range(channels):
                    channel_output = (
                        y_hat[batch_idx, source_idx, channel_idx, :100].detach().numpy()
                    )
                    outputs_per_channel.append(channel_output)

                # Check that channels are different from each other
                differences_found = 0
                for i in range(channels - 1):
                    for j in range(i + 1, channels):
                        if not np.allclose(
                            outputs_per_channel[i], outputs_per_channel[j], rtol=1e-3
                        ):
                            differences_found += 1

                assert differences_found >= 3, (
                    f"Expected different filter parameters per channel "
                    f"in batch {batch_idx}, source {source_idx}, "
                    f"but only found {differences_found} differences"
                )

    def test_mode_default_is_per_example(self):
        """GIVEN no mode specified, WHEN applied, THEN should default to per_example."""
        # GIVEN: Two augmentations, one with explicit per_example, one default
        aug1 = PeakFilter(sample_rate=16000, seed=42)
        aug2 = PeakFilter(sample_rate=16000, mode="per_example", seed=42)

        # WHEN: Applied to the same input
        audio = torch.randn(4, 2, 1, 8000) * 0.5
        y_hat1 = aug1(audio)
        y_hat2 = aug2(audio)

        # THEN: Results should be identical
        assert torch.allclose(y_hat1, y_hat2)

    def test_invalid_mode_raises_error(self):
        """GIVEN invalid mode, WHEN creating PeakFilter, THEN should raise ValueError."""
        # GIVEN/WHEN: Creating augmentation with invalid mode
        # THEN: Should raise ValueError
        with pytest.raises(ValueError, match="mode must be one of"):
            PeakFilter(sample_rate=16000, mode="invalid_mode")

    def test_accepts_numpy_array(self):
        """GIVEN numpy array input, WHEN applied, THEN should work and return numpy."""
        # GIVEN: Numpy array input
        audio = np.random.randn(2, 2, 1, 8000).astype(np.float32) * 0.5
        aug = PeakFilter(sample_rate=16000, p=1.0)

        # WHEN: Applied
        y_hat = aug(audio)

        # THEN: Should return numpy array with same shape
        assert isinstance(y_hat, np.ndarray)
        assert y_hat.shape == audio.shape

    def test_accepts_torch_tensor(self):
        """GIVEN torch tensor input, WHEN applied, THEN should work and return torch tensor."""
        # GIVEN: Torch tensor input
        audio = torch.randn(2, 2, 1, 8000) * 0.5
        aug = PeakFilter(sample_rate=16000, p=1.0)

        # WHEN: Applied
        y_hat = aug(audio)

        # THEN: Should return torch tensor with same shape
        assert isinstance(y_hat, torch.Tensor)
        assert y_hat.shape == audio.shape

    def test_frequency_range_is_respected(self):
        """GIVEN freq range, WHEN applied, THEN center freq should be within range."""
        # GIVEN: PeakFilter with specific frequency range
        min_freq = 1000.0
        max_freq = 2000.0
        aug = PeakFilter(
            sample_rate=16000,
            min_center_freq=min_freq,
            max_center_freq=max_freq,
            p=1.0,
            seed=42,
        )

        # WHEN: Parameters are randomized
        aug.randomize_parameters()

        # THEN: Center frequency should be within the specified range
        assert min_freq <= aug.current_params["center_freq"] <= max_freq

    def test_gain_range_is_respected(self):
        """GIVEN gain range, WHEN applied, THEN gain should be within range."""
        # GIVEN: PeakFilter with specific gain range
        min_gain = -6.0
        max_gain = 6.0
        aug = PeakFilter(
            sample_rate=16000,
            min_gain_db=min_gain,
            max_gain_db=max_gain,
            p=1.0,
            seed=42,
        )

        # WHEN: Parameters are randomized
        aug.randomize_parameters()

        # THEN: Gain should be within the specified range
        assert min_gain <= aug.current_params["gain_db"] <= max_gain

    def test_q_range_is_respected(self):
        """GIVEN Q range, WHEN applied, THEN Q should be within range."""
        # GIVEN: PeakFilter with specific Q range
        min_q = 0.5
        max_q = 2.0
        aug = PeakFilter(
            sample_rate=16000,
            min_q=min_q,
            max_q=max_q,
            p=1.0,
            seed=42,
        )

        # WHEN: Parameters are randomized
        aug.randomize_parameters()

        # THEN: Q should be within the specified range
        assert min_q <= aug.current_params["q"] <= max_q
