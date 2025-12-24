"""Pytest configuration and shared fixtures."""

import pytest
import torch

from auementations.core.base import BaseAugmentation


class MockAugmentation(BaseAugmentation):
    """Mock augmentation for testing."""

    def __init__(
        self,
        sample_rate: int,
        gain: float = 1.0,
        p: float = 1.0,
        seed: int = None,
    ):
        super().__init__(sample_rate=sample_rate, p=p, seed=seed)
        self.gain = gain
        self.apply_count = 0

    def forward(self, audio):
        self.apply_count += 1
        return audio * self.gain

    def to_config(self):
        config = super().to_config()
        config["gain"] = self.gain
        return config


@pytest.fixture
def sample_rate():
    """Standard sample rate for testing."""
    return 16000


@pytest.fixture
def mono_audio():
    """Generate mono audio signal as torch tensor with shape (source, channel, time)."""
    duration = 1.0  # seconds
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    # Generate a simple sine wave
    t = torch.linspace(0, duration, num_samples)
    frequency = 440.0  # Hz (A4 note)
    audio = torch.sin(2 * torch.pi * frequency * t).to(torch.float32)
    # Shape: (1 source, 1 channel, num_samples time)
    return audio.unsqueeze(0).unsqueeze(0)


@pytest.fixture
def mono_audio_with_batch(mono_audio):
    """Generate batch of mono audio with shape (batch, source, channel, time)."""
    # mono_audio is (1, 1, time), add batch dimension
    return mono_audio.unsqueeze(0)


@pytest.fixture
def stereo_audio():
    """Generate stereo audio signal as torch tensor with shape (source, channel, time)."""
    duration = 1.0  # seconds
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    # Generate two slightly different sine waves
    t = torch.linspace(0, duration, num_samples)
    left = torch.sin(2 * torch.pi * 440.0 * t).to(torch.float32)
    right = torch.sin(2 * torch.pi * 445.0 * t).to(torch.float32)
    # Shape: (1 source, 2 channels, num_samples time)
    stereo = torch.stack([left, right], dim=0)  # (2, time)
    return stereo.unsqueeze(0)  # (1, 2, time)


@pytest.fixture
def mock_augmentation(sample_rate):
    """Create a mock augmentation for testing."""
    return MockAugmentation(sample_rate=sample_rate, gain=2.0)
