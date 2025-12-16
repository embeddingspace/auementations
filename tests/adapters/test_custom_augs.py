from contextlib import nullcontext

import pytest
import torch
from hydra_zen import instantiate

from auementations.adapters.custom import GainAugmentation, NoiseAugmentation
from auementations.config.config_store import auementations_store
from auementations.utils import amplitude_to_db


def has_uniform_gain(
    tensor: torch.Tensor, tol: float = 1e-5, epsilon: float = 1e-10
) -> bool:
    """Helper function to assert that a tensor has uniform values (constant gain applied).

    Checks that the standard deviation is very small relative to the mean,
    which is more robust than allclose for ratio comparisons.
    """
    std_dev = tensor.std()
    mean_val = tensor.mean()
    relative_std = std_dev / (mean_val.abs() + epsilon)  # Avoid division by zero

    return (relative_std < tol).item()


class TestGainAugmentation:
    @pytest.mark.parametrize(
        "sample_rate,expectation",
        [
            (0, pytest.raises(ValueError)),
            (16000, nullcontext()),
            (16000.0, nullcontext()),
        ],
    )
    def test_should_accept_sample_rate_arg(self, sample_rate, expectation):
        """Must accept sample_rate to comply with spec."""
        with expectation:
            aug = GainAugmentation(sample_rate=sample_rate)
            assert aug.sample_rate == sample_rate

    def test_should_instantiate_with_default_params(self):
        aug = GainAugmentation()
        assert aug is not None

    def test_forward_pass_should_return_identical_signal_with_different_gain(self):
        sig = torch.clip(torch.rand(1, 1000) * 2 - 1, -1, 1)
        aug = GainAugmentation(min_gain_db=-3, max_gain_db=0, p=1.0)
        y_hat = aug(sig)

        ratio = sig / y_hat
        assert torch.isclose((y_hat * ratio), sig).all()

    def test_should_never_return_signal_gt_one(self):
        sig = torch.clip((torch.rand(1, 1000) * 2 - 1) * 0.95, -1, 1)
        aug = GainAugmentation(min_gain_db=0, max_gain_db=6, p=1.0)

        for _ in range(10):
            y_hat = aug(sig)
            assert (y_hat < 1.0).all()

    def test_p0_should_always_return_original(self):
        sig = torch.clip((torch.rand(1, 1000) * 2 - 1) * 0.95, -1, 1)
        aug = GainAugmentation(min_gain_db=-6, max_gain_db=0, p=0.0)

        for _ in range(3):
            y_hat = aug(sig)
            assert torch.isclose(y_hat, sig).all()

    def test_p1_should_never_return_original(self):
        sig = torch.clip((torch.rand(1, 1000) * 2 - 1) * 0.95, -1, 1)
        aug = GainAugmentation(min_gain_db=-12, max_gain_db=-6, p=1.0)

        for _ in range(3):
            y_hat = aug(sig)
            assert not torch.isclose(y_hat, sig).all()

    # Mode parameter tests
    def test_mode_per_batch_applies_same_gain_to_entire_batch(self):
        # GIVEN: A batch of audio with shape (batch, channels, samples)
        batch_size = 4
        channels = 2
        samples = 1000
        sig = (torch.rand(batch_size, channels, samples) * 2 - 1) * 0.5
        aug = GainAugmentation(
            min_gain_db=-6, max_gain_db=6, p=1.0, mode="per_batch", seed=42
        )

        # WHEN: The augmentation is applied
        y_hat = aug(sig)

        # THEN: All examples receive the same gain
        # Check that the ratio between output and input is constant across all examples
        ratios = y_hat / (sig + 1e-8)
        first_example_ratio = ratios[0].mean()
        for i in range(1, batch_size):
            assert torch.isclose(ratios[i].mean(), first_example_ratio, rtol=1e-4), (
                f"Example {i} has different gain than example 0"
            )

    def test_mode_per_example_applies_different_gain_per_example(self):
        # GIVEN: A batch of audio with shape (batch, channels, samples)
        batch_size = 4
        channels = 2
        samples = 1000
        sig = torch.clip(
            (torch.rand(batch_size, channels, samples) * 2 - 1) * 0.5, -1, 1
        )
        aug = GainAugmentation(
            min_gain_db=-6, max_gain_db=6, p=1.0, mode="per_example", seed=42
        )

        # WHEN: The augmentation is applied
        y_hat = aug(sig)

        # THEN: Each example receives a different gain
        # Compute the gain applied to each example
        gains = []
        for i in range(batch_size):
            ratio = y_hat[i] / (sig[i] + 1e-8)
            gains.append(ratio.mean().item())

        # Check that at least some gains are different (with high probability)
        unique_gains = len(set([round(g, 4) for g in gains]))
        assert unique_gains > 1, (
            "All examples received the same gain in per_example mode"
        )

        # Each example should have uniform gain across its channels and samples
        for i in range(batch_size):
            ratio = y_hat[i] / (sig[i] + 1e-8)
            assert has_uniform_gain(ratio, tol=1e-4), (
                f"Example {i} does not have uniform gain across channels/samples"
            )

    def test_mode_per_source_applies_different_gain_per_source(self):
        # GIVEN: Audio with sources in dimension 1, shape (batch, sources, samples)
        batch_size = 2
        sources = 4
        samples = 1000
        sig = torch.clip(
            (torch.rand(batch_size, sources, samples) * 2 - 1) * 0.5, -1, 1
        )
        aug = GainAugmentation(
            min_gain_db=-6, max_gain_db=6, p=1.0, mode="per_source", seed=42
        )

        # WHEN: The augmentation is applied
        y_hat = aug(sig)

        # THEN: Each source receives a different gain (within each example)
        for batch_idx in range(batch_size):
            gains = []
            for source_idx in range(sources):
                ratio = y_hat[batch_idx, source_idx] / (
                    sig[batch_idx, source_idx] + 1e-8
                )
                gains.append(ratio.mean().item())

            # Check that at least some gains are different
            unique_gains = len(set([round(g, 4) for g in gains]))
            assert unique_gains > 1, (
                f"All sources in example {batch_idx} received the same gain"
            )

            # Each source should have uniform gain across its samples
            for source_idx in range(sources):
                ratio = y_hat[batch_idx, source_idx] / (
                    sig[batch_idx, source_idx] + 1e-8
                )
                assert has_uniform_gain(ratio, tol=1e-4), (
                    f"Source {source_idx} in example {batch_idx} does not have uniform gain",
                )

    def test_mode_per_channel_applies_different_gain_per_channel(self):
        # GIVEN: Audio with channels in dimension 2, shape (batch, sources, channels, samples)
        batch_size = 2
        sources = 2
        channels = 4
        samples = 1000
        sig = torch.clip(
            (torch.rand(batch_size, sources, channels, samples) * 2 - 1) * 0.5, -1, 1
        )
        aug = GainAugmentation(
            min_gain_db=-6, max_gain_db=6, p=1.0, mode="per_channel", seed=42
        )

        # WHEN: The augmentation is applied
        y_hat = aug(sig)

        # THEN: Each channel receives a different gain
        for batch_idx in range(batch_size):
            for source_idx in range(sources):
                gains = []
                for channel_idx in range(channels):
                    ratio = y_hat[batch_idx, source_idx, channel_idx] / (
                        sig[batch_idx, source_idx, channel_idx] + 1e-8
                    )
                    gains.append(ratio.mean().item())

                # Check that at least some gains are different
                unique_gains = len(set([round(g, 4) for g in gains]))
                assert unique_gains > 1, (
                    f"All channels in example {batch_idx}, source {source_idx} received the same gain"
                )

                # Each channel should have uniform gain across its samples
                for channel_idx in range(channels):
                    ratio = y_hat[batch_idx, source_idx, channel_idx] / (
                        sig[batch_idx, source_idx, channel_idx] + 1e-8
                    )
                    assert has_uniform_gain(ratio, tol=5e-4), (
                        f"Channel {channel_idx} does not have uniform gain",
                    )

    def test_mode_default_is_per_example(self):
        # GIVEN: An augmentation without specifying mode
        aug1 = GainAugmentation(min_gain_db=-6, max_gain_db=6, p=1.0, seed=42)
        aug2 = GainAugmentation(
            min_gain_db=-6, max_gain_db=6, p=1.0, mode="per_example", seed=42
        )

        # WHEN: Applied to the same input
        sig = torch.clip((torch.rand(4, 2, 1000) * 2 - 1) * 0.5, -1, 1)
        y_hat1 = aug1(sig)
        y_hat2 = aug2(sig)

        # THEN: Results should be identical
        assert torch.allclose(y_hat1, y_hat2)

    def test_invalid_mode_raises_error(self):
        # GIVEN/WHEN: Creating augmentation with invalid mode
        # THEN: Should raise ValueError
        with pytest.raises(ValueError, match="mode must be one of"):
            GainAugmentation(mode="invalid_mode")

    def test_mode_per_example_with_2d_input(self):
        # GIVEN: A 2D input (batch, samples) - common for single channel
        batch_size = 4
        samples = 1000
        sig = (torch.rand(batch_size, samples) * 2 - 1) * 0.5
        aug = GainAugmentation(
            min_gain_db=-6, max_gain_db=6, p=1.0, mode="per_example", seed=42
        )

        # WHEN: The augmentation is applied
        y_hat = aug(sig)

        # THEN: Each example receives a different gain
        gains = []
        for i in range(batch_size):
            ratio = y_hat[i] / (sig[i] + 1e-8)
            gains.append(ratio.mean().item())

        unique_gains = len(set([round(g, 4) for g in gains]))
        assert unique_gains > 1, "All examples received the same gain"

    def test_mode_per_source_with_3d_input(self):
        # GIVEN: A 3D input (batch, sources, samples)
        batch_size = 2
        sources = 3
        samples = 1000
        sig = (torch.rand(batch_size, sources, samples) * 2 - 1) * 0.5
        aug = GainAugmentation(
            min_gain_db=-6, max_gain_db=6, p=1.0, mode="per_source", seed=42
        )

        # WHEN: The augmentation is applied
        y_hat = aug(sig)

        # THEN: Shape is preserved and each source has different gain
        assert y_hat.shape == sig.shape
        for batch_idx in range(batch_size):
            gains = []
            for source_idx in range(sources):
                ratio = y_hat[batch_idx, source_idx] / (
                    sig[batch_idx, source_idx] + 1e-8
                )
                gains.append(ratio.mean().item())
            unique_gains = len(set([round(g, 4) for g in gains]))
            assert unique_gains > 1


class TestNoise:
    @pytest.mark.parametrize("db_gain", [-100.0, -80.0, -60.0])
    def test_single_value_adds_constant_noise_level_to_input(self, db_gain):
        aug = NoiseAugmentation(gain_db=db_gain)
        signal = torch.zeros(1, 1, 1, 1200)

        y_hat = aug(signal)
        noise_gain = torch.abs(y_hat).max()
        noise_gain_db = amplitude_to_db(noise_gain)

        assert torch.isclose(torch.as_tensor(db_gain), noise_gain_db, rtol=1e-3)

    @pytest.mark.parametrize("amp_gain", [1.0e-8, 1.0e-6, 1.0e-4])
    def test_single_value_amplitude_adds_constant_noise_level_to_input(self, amp_gain):
        aug = NoiseAugmentation(gain_amp=amp_gain)
        signal = torch.zeros(1, 1, 1, 1200)

        y_hat = aug(signal)
        noise_gain = torch.abs(y_hat).max()

        assert (noise_gain - amp_gain).abs() < 1e-6

    def test_db_range_batch_mode_produces_noise_no_larger_than_max(self):
        db_range_min, db_range_max = torch.tensor(-100.0), torch.tensor(-80.0)
        aug = NoiseAugmentation(min_gain_db=db_range_min, max_gain_db=db_range_max)
        signal = torch.zeros(1, 1, 1, 1200)

        y_hat = aug(signal)
        noise_gain = torch.abs(y_hat).max()
        noise_max_db = amplitude_to_db(noise_gain)

        # < 3dB difference
        assert (db_range_max - noise_max_db).abs() < 3

    def test_db_range_example_mode_produces_different_gains_per_example(self):
        db_range_min, db_range_max = torch.tensor(-100.0), torch.tensor(-80.0)
        aug = NoiseAugmentation(
            min_gain_db=db_range_min, max_gain_db=db_range_max, mode="per_example"
        )
        signal = torch.zeros(8, 1, 1, 1200)

        y_hat = aug(signal)
        noise_gain = torch.abs(y_hat).amax(-1)
        noise_max_db = amplitude_to_db(noise_gain).flatten()

        # Compare the first one against all the others;
        # they should be mostly different.
        noise_db_0 = noise_max_db[0]
        for noise_gain in noise_max_db[1:]:
            # we should see no differences < .2dB.
            assert (noise_db_0 - noise_gain).abs() > 0.2


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_handle2_3_4_dims(ndim):
    batch_size = 2
    sources = 1
    channels = 1
    samples = 1000
    sample_rate = 16000

    config = auementations_store.get_entry(group="auementations", name="gain")["node"]

    match ndim:
        case 3:
            shape = (batch_size, samples)
        case 4:
            shape = (batch_size, channels, samples)
        case _:
            shape = (batch_size, sources, channels, samples)

    audio = (torch.rand(*shape) * 2 - 1) * 0.5
    augmentation = instantiate(config, sample_rate=sample_rate, p=1.0)

    y_hat = augmentation(audio)
    assert audio.shape == y_hat.shape
