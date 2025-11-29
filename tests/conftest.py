"""Pytest configuration and shared fixtures."""

import numpy as np
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

    def __call__(self, audio, **kwargs):
        if not self.should_apply():
            return audio
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
    """Generate mono audio signal (1D array)."""
    duration = 1.0  # seconds
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    # Generate a simple sine wave
    t = np.linspace(0, duration, num_samples)
    frequency = 440.0  # Hz (A4 note)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture
def mono_audio_with_batch(mono_audio):
    return torch.tensor(mono_audio).unsqueeze(0)


@pytest.fixture
def stereo_audio():
    """Generate stereo audio signal (2D array)."""
    duration = 1.0  # seconds
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    # Generate two slightly different sine waves
    t = np.linspace(0, duration, num_samples)
    left = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 445.0 * t).astype(np.float32)
    return np.stack([left, right])


@pytest.fixture
def mock_augmentation(sample_rate):
    """Create a mock augmentation for testing."""
    return MockAugmentation(sample_rate=sample_rate, gain=2.0)
