"""Parametrized tests to validate output types for all augmentations.

These tests ensure that each augmentation:
1. Returns torch.Tensor when given torch.Tensor input
2. Returns numpy.ndarray when given numpy.ndarray input
3. Preserves the input type consistently
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pedalboard")

from auementations.adapters.pedalboard import (  # noqa: E402
    HighPassFilter as PedalboardHighPassFilter,
)
from auementations.adapters.pedalboard import (  # noqa: E402
    LowPassFilter as PedalboardLowPassFilter,
)
from auementations.adapters.torch_audiomentations import (  # noqa: E402
    AddColoredNoise,
    Gain,
    PitchShift,
)
from auementations.adapters.torch_audiomentations import (  # noqa: E402
    HighPassFilter as TorchHighPassFilter,
)
from auementations.adapters.torch_audiomentations import (  # noqa: E402
    LowPassFilter as TorchLowPassFilter,
)

# Define all augmentations to test with their initialization parameters
TORCH_AUDIOMENTATIONS = [
    (
        "Gain",
        Gain,
        {"sample_rate": 16000, "min_gain_db": -6.0, "max_gain_db": 6.0, "p": 1.0},
    ),
    (
        "PitchShift",
        PitchShift,
        {"sample_rate": 16000, "min_semitones": -2.0, "max_semitones": 2.0, "p": 1.0},
    ),
    (
        "AddColoredNoise",
        AddColoredNoise,
        {
            "sample_rate": 16000,
            "min_snr_db": 10.0,
            "max_snr_db": 30.0,
            "min_f_decay": -2.0,
            "max_f_decay": 2.0,
            "p": 1.0,
        },
    ),
    (
        "TorchHighPassFilter",
        TorchHighPassFilter,
        {
            "sample_rate": 16000,
            "min_cutoff_freq": 100.0,
            "max_cutoff_freq": 500.0,
            "p": 1.0,
        },
    ),
    (
        "TorchLowPassFilter",
        TorchLowPassFilter,
        {
            "sample_rate": 16000,
            "min_cutoff_freq": 1000.0,
            "max_cutoff_freq": 5000.0,
            "p": 1.0,
        },
    ),
]

PEDALBOARD_AUGMENTATIONS = [
    (
        "PedalboardLowPassFilter",
        PedalboardLowPassFilter,
        {
            "sample_rate": 16000,
            "min_cutoff_freq": 500.0,
            "max_cutoff_freq": 2000.0,
            "p": 1.0,
        },
    ),
    (
        "PedalboardHighPassFilter",
        PedalboardHighPassFilter,
        {
            "sample_rate": 16000,
            "min_cutoff_freq": 100.0,
            "max_cutoff_freq": 500.0,
            "p": 1.0,
        },
    ),
]

ALL_AUGMENTATIONS = TORCH_AUDIOMENTATIONS + PEDALBOARD_AUGMENTATIONS


class TestTorchAudiomentationsOutputTypes:
    """Test that torch_audiomentations augmentations preserve tensor types."""

    @pytest.mark.parametrize("name,aug_class,params", TORCH_AUDIOMENTATIONS)
    def test_given_torch_tensor_when_augmented_then_returns_torch_tensor(
        self, name, aug_class, params
    ):
        """Given torch.Tensor input, when augmentation applied, then returns torch.Tensor."""
        # GIVEN: A torch tensor and an augmentation
        audio = torch.randn(1, 1, 16000)
        augmentation = aug_class(**params)

        # WHEN: Apply augmentation
        result = augmentation(audio)

        # THEN: Output is a torch.Tensor
        assert isinstance(result, torch.Tensor), (
            f"{name} did not return torch.Tensor for torch.Tensor input"
        )
        assert result.shape == audio.shape

    @pytest.mark.parametrize("name,aug_class,params", TORCH_AUDIOMENTATIONS)
    def test_given_numpy_array_when_augmented_then_returns_numpy_array(
        self, name, aug_class, params
    ):
        """Given numpy.ndarray input, when augmentation applied, then returns numpy.ndarray."""
        # GIVEN: A numpy array and an augmentation
        audio = np.random.randn(1, 1, 16000).astype(np.float32)
        augmentation = aug_class(**params)

        # WHEN: Apply augmentation
        result = augmentation(audio)

        # THEN: Output is a numpy.ndarray
        assert isinstance(result, np.ndarray), (
            f"{name} did not return numpy.ndarray for numpy.ndarray input"
        )
        assert result.shape == audio.shape

    @pytest.mark.parametrize("name,aug_class,params", TORCH_AUDIOMENTATIONS)
    def test_given_mono_torch_tensor_when_augmented_then_returns_torch_tensor(
        self, name, aug_class, params
    ):
        """Given mono torch.Tensor, when augmentation applied, then returns torch.Tensor."""
        # GIVEN: A mono torch tensor (1D)
        audio = torch.randn(16000)
        augmentation = aug_class(**params)

        # WHEN: Apply augmentation
        result = augmentation(audio)

        # THEN: Output is a torch.Tensor with same shape
        assert isinstance(result, torch.Tensor), (
            f"{name} did not return torch.Tensor for mono torch.Tensor input"
        )
        assert result.shape == audio.shape

    @pytest.mark.parametrize("name,aug_class,params", TORCH_AUDIOMENTATIONS)
    def test_given_stereo_torch_tensor_when_augmented_then_returns_torch_tensor(
        self, name, aug_class, params
    ):
        """Given stereo torch.Tensor, when augmentation applied, then returns torch.Tensor."""
        # GIVEN: A stereo torch tensor (2, samples)
        audio = torch.randn(2, 16000)
        augmentation = aug_class(**params)

        # WHEN: Apply augmentation
        result = augmentation(audio)

        # THEN: Output is a torch.Tensor with same shape
        assert isinstance(result, torch.Tensor), (
            f"{name} did not return torch.Tensor for stereo torch.Tensor input"
        )
        assert result.shape == audio.shape


class TestPedalboardOutputTypes:
    """Test that pedalboard augmentations handle input/output types correctly."""

    @pytest.mark.parametrize("name,aug_class,params", PEDALBOARD_AUGMENTATIONS)
    def test_given_numpy_array_when_augmented_then_returns_numpy_array(
        self, name, aug_class, params
    ):
        """Given numpy.ndarray input, when augmentation applied, then returns numpy.ndarray."""
        # GIVEN: A numpy array and a pedalboard augmentation
        audio = np.random.randn(16000).astype(np.float32)
        augmentation = aug_class(**params)

        # WHEN: Apply augmentation
        result = augmentation(audio)

        # THEN: Output is a numpy.ndarray
        assert isinstance(result, np.ndarray), (
            f"{name} did not return numpy.ndarray for numpy.ndarray input"
        )
        assert result.shape == audio.shape
        assert result.dtype == audio.dtype

    @pytest.mark.parametrize("name,aug_class,params", PEDALBOARD_AUGMENTATIONS)
    def test_given_torch_tensor_when_augmented_then_returns_torch_tensor(
        self, name, aug_class, params
    ):
        """Given torch.Tensor input, when augmentation applied, then returns torch.Tensor."""
        # GIVEN: A torch tensor and a pedalboard augmentation
        audio = torch.randn(16000)
        augmentation = aug_class(**params)

        # WHEN: Apply augmentation
        result = augmentation(audio)

        # THEN: Output is a torch.Tensor
        assert isinstance(result, torch.Tensor), (
            f"{name} did not return torch.Tensor for torch.Tensor input"
        )
        assert result.shape == audio.shape

    @pytest.mark.parametrize("name,aug_class,params", PEDALBOARD_AUGMENTATIONS)
    def test_given_stereo_numpy_when_augmented_then_returns_numpy_array(
        self, name, aug_class, params
    ):
        """Given stereo numpy.ndarray, when augmentation applied, then returns numpy.ndarray."""
        # GIVEN: A stereo numpy array (2, samples)
        audio = np.random.randn(2, 16000).astype(np.float32)
        augmentation = aug_class(**params)

        # WHEN: Apply augmentation
        result = augmentation(audio)

        # THEN: Output is a numpy.ndarray with same shape
        assert isinstance(result, np.ndarray), (
            f"{name} did not return numpy.ndarray for stereo numpy.ndarray input"
        )
        assert result.shape == audio.shape
        assert result.dtype == audio.dtype

    @pytest.mark.parametrize("name,aug_class,params", PEDALBOARD_AUGMENTATIONS)
    def test_given_stereo_torch_tensor_when_augmented_then_returns_torch_tensor(
        self, name, aug_class, params
    ):
        """Given stereo torch.Tensor, when augmentation applied, then returns torch.Tensor."""
        # GIVEN: A stereo torch tensor (2, samples)
        audio = torch.randn(2, 16000)
        augmentation = aug_class(**params)

        # WHEN: Apply augmentation
        result = augmentation(audio)

        # THEN: Output is a torch.Tensor with same shape
        assert isinstance(result, torch.Tensor), (
            f"{name} did not return torch.Tensor for stereo torch.Tensor input"
        )
        assert result.shape == audio.shape

    @pytest.mark.parametrize("name,aug_class,params", PEDALBOARD_AUGMENTATIONS)
    def test_given_different_dtypes_when_augmented_then_preserves_dtype(
        self, name, aug_class, params
    ):
        """Given different dtypes, when augmentation applied, then preserves dtype."""
        # GIVEN: Numpy arrays with different dtypes
        augmentation = aug_class(**params)

        for dtype in [np.float32, np.float64]:
            audio = np.random.randn(16000).astype(dtype)

            # WHEN: Apply augmentation
            result = augmentation(audio)

            # THEN: Output dtype matches input dtype
            assert result.dtype == dtype, (
                f"{name} did not preserve dtype {dtype}, got {result.dtype}"
            )

    @pytest.mark.parametrize("name,aug_class,params", PEDALBOARD_AUGMENTATIONS)
    def test_given_torch_cuda_tensor_when_augmented_then_returns_torch_cuda_tensor(
        self, name, aug_class, params
    ):
        """Given CUDA torch.Tensor, when augmentation applied, then returns CUDA torch.Tensor."""
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # GIVEN: A CUDA torch tensor and a pedalboard augmentation
        audio = torch.randn(16000).cuda()
        augmentation = aug_class(**params)

        # WHEN: Apply augmentation
        result = augmentation(audio)

        # THEN: Output is a torch.Tensor on the same device
        assert isinstance(result, torch.Tensor), (
            f"{name} did not return torch.Tensor for CUDA torch.Tensor input"
        )
        assert result.device.type == "cuda", f"{name} did not preserve CUDA device"
        assert result.shape == audio.shape


class TestAllAugmentationsConsistency:
    """Test consistency across all augmentations regardless of backend."""

    @pytest.mark.parametrize("name,aug_class,params", ALL_AUGMENTATIONS)
    def test_given_augmentation_with_p_zero_when_applied_then_returns_unchanged(
        self, name, aug_class, params
    ):
        """Given p=0.0, when augmentation applied, then returns input unchanged."""
        # GIVEN: An augmentation with p=0.0 (never applies)
        params_copy = params.copy()
        params_copy["p"] = 0.0
        augmentation = aug_class(**params_copy)

        # Test with numpy
        audio_np = np.random.randn(16000).astype(np.float32)
        result_np = augmentation(audio_np)
        assert np.array_equal(result_np, audio_np), (
            f"{name} with p=0.0 modified numpy input"
        )

        # Test with torch (all augmentations now support torch tensors)
        audio_torch = torch.randn(16000)
        result_torch = augmentation(audio_torch)
        assert torch.equal(result_torch, audio_torch), (
            f"{name} with p=0.0 modified torch input"
        )

    @pytest.mark.parametrize("name,aug_class,params", ALL_AUGMENTATIONS)
    def test_given_augmentation_when_applied_multiple_times_then_produces_variation(
        self, name, aug_class, params
    ):
        """Given an augmentation, when applied multiple times, then produces variation due to randomness."""
        # GIVEN: An augmentation with range parameters
        augmentation = aug_class(**params)
        audio = np.random.randn(16000).astype(np.float32)

        # WHEN: Apply multiple times
        results = [augmentation(audio.copy()) for _ in range(10)]

        # THEN: At least some results should differ (due to parameter randomization)
        unique_results = []
        for result in results:
            is_unique = True
            for unique in unique_results:
                if np.allclose(result, unique, atol=1e-5):
                    is_unique = False
                    break
            if is_unique:
                unique_results.append(result)

        # Should have some variation (at least 2 unique results from 10 applications)
        assert len(unique_results) >= 2, (
            f"{name} produced no variation across 10 applications"
        )

    @pytest.mark.parametrize("name,aug_class,params", ALL_AUGMENTATIONS)
    def test_given_augmentation_when_called_then_output_not_nan_or_inf(
        self, name, aug_class, params
    ):
        """Given an augmentation, when applied, then output contains no NaN or Inf values."""
        # GIVEN: An augmentation and valid audio
        augmentation = aug_class(**params)
        audio = np.random.randn(16000).astype(np.float32) * 0.5  # Reasonable amplitude

        # WHEN: Apply augmentation
        result = augmentation(audio)

        # THEN: Result contains no NaN or Inf
        if isinstance(result, torch.Tensor):
            assert not torch.isnan(result).any(), f"{name} produced NaN values"
            assert not torch.isinf(result).any(), f"{name} produced Inf values"
        else:
            assert not np.isnan(result).any(), f"{name} produced NaN values"
            assert not np.isinf(result).any(), f"{name} produced Inf values"
