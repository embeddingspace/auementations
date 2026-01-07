"""Tests for augmentation logging functionality."""

import pytest
import torch

from auementations.adapters.custom import GainAugmentation, NoiseAugmentation
from auementations.adapters.pedalboard import LowPassFilter, PeakFilter
from auementations.core.composition import Compose, OneOf, SomeOf


@pytest.mark.parametrize(
    "aug_cls,aug_kwargs",
    [
        (GainAugmentation, dict(min_gain_db=-6.0, max_gain_db=6.0)),
        (NoiseAugmentation, dict(min_gain_db=-6.0, max_gain_db=6.0)),
        (LowPassFilter, dict(min_cutoff_freq=80, max_cutoff_freq=200)),
    ],
)
class TestSimpleAugmentationLogging:
    """Tests for logging in simple augmentations."""

    def test_call_without_log_returns_only_audio(self, aug_cls, aug_kwargs):
        # GIVEN: A simple augmentation
        aug = aug_cls(p=1.0, seed=42, sample_rate=16000, **aug_kwargs)
        audio = torch.randn(1, 1000)

        # WHEN: Called without log parameter
        result = aug(audio)

        # THEN: Should return only audio tensor (not tuple)
        assert isinstance(result, torch.Tensor)
        assert result.shape == audio.shape

    def test_call_with_log_false_returns_only_audio(self, aug_cls, aug_kwargs):
        # GIVEN: A simple augmentation
        aug = aug_cls(p=1.0, seed=42, sample_rate=16000, **aug_kwargs)
        audio = torch.randn(1, 1000)

        # WHEN: Called with log=False
        result = aug(audio, log=False)

        # THEN: Should return only audio tensor (not tuple)
        assert isinstance(result, torch.Tensor)
        assert result.shape == audio.shape

    def test_call_with_log_true_returns_audio_and_dict(self, aug_cls, aug_kwargs):
        # GIVEN: A simple augmentation in per_batch mode (single dict output)
        aug = aug_cls(**aug_kwargs, p=1.0, sample_rate=16000, mode="per_batch", seed=42)
        audio = torch.randn(1, 1000)

        # WHEN: Called with log=True
        result = aug(audio, log=True)

        # THEN: Should return tuple of (audio, log_dict)
        assert isinstance(result, tuple)
        assert len(result) == 2
        audio_out, log_dict = result
        assert isinstance(audio_out, torch.Tensor)
        assert isinstance(log_dict, dict)

    def test_log_dict_contains_augmentation_name_and_parameters(
        self, aug_cls, aug_kwargs
    ):
        # GIVEN: A simple augmentation with fixed parameters in per_batch mode
        aug = aug_cls(**aug_kwargs, sample_rate=16000, p=1.0, mode="per_batch", seed=42)
        audio = torch.randn(1, 1000)

        # WHEN: Called with log=True
        audio_out, log_dict = aug(audio, log=True)

        # THEN: Log dict should contain augmentation name and parameters
        assert "augmentation" in log_dict
        assert log_dict["augmentation"] == aug_cls.__name__
        assert "parameters" in log_dict
        assert isinstance(log_dict["parameters"], dict)
        # Should contain the actual sampled gain value
        # TODO add verification of parameter names
        # assert "gain_db" in log_dict["parameters"]

    @pytest.mark.xfail(reason="not all aug_cls use gain_db")
    def test_log_contains_actual_sampled_parameters(self, aug_cls, aug_kwargs):
        # GIVEN: An augmentation with a range parameter in per_batch mode
        aug = aug_cls(**aug_kwargs, sample_rate=16000, p=1.0, mode="per_batch", seed=42)
        audio = torch.randn(1, 1000)

        # WHEN: Called multiple times with log=True
        _, log1 = aug(audio, log=True)
        _, log2 = aug(audio, log=True)

        # THEN: Each log should contain the actual sampled parameter value
        gain1 = log1["parameters"]["gain_db"]
        gain2 = log2["parameters"]["gain_db"]
        # With different random states, gains should likely be different
        # (though this is probabilistic, seed ensures first call is deterministic)
        assert isinstance(gain1, (int, float))
        assert isinstance(gain2, (int, float))
        # Note: GainAugmentation limits gain based on signal level to prevent clipping
        # We just verify that we get numeric values, not specific ranges

    def test_log_is_none_when_augmentation_not_applied(self, aug_cls, aug_kwargs):
        # GIVEN: An augmentation with p=0 (never applies)
        aug = aug_cls(
            **aug_kwargs, sample_rate=16000, p=0.0, seed=42, mode="per_example"
        )
        audio = torch.randn(1, 1000)

        # WHEN: Called with log=True
        audio_out, log_dict = aug(audio, log=True)

        # THEN: Log should be None
        assert log_dict == [None]
        # Audio should be unchanged
        assert torch.allclose(audio_out, audio)

    def test_log_is_present_when_augmentation_applied_with_probability(
        self, aug_cls, aug_kwargs
    ):
        # GIVEN: An augmentation with p=1.0 (always applies) in per_batch mode
        aug = aug_cls(**aug_kwargs, sample_rate=16000, p=1.0, mode="per_batch", seed=42)
        audio = torch.randn(1, 1000)

        # WHEN: Called with log=True
        audio_out, log_dict = aug(audio, log=True)

        # THEN: Log should be a dict (not None)
        assert log_dict is not None
        assert isinstance(log_dict, dict)
        assert "augmentation" in log_dict
        assert "parameters" in log_dict


@pytest.mark.parametrize("aug_cls", [GainAugmentation, NoiseAugmentation])
class TestPerExampleLogging:
    """Tests for logging in per-example mode."""

    def test_per_example_mode_returns_list_of_logs(self, aug_cls):
        # GIVEN: An augmentation in per_example mode with a batch
        batch_size = 4
        aug = aug_cls(
            min_gain_db=-6.0, max_gain_db=6.0, p=1.0, mode="per_example", seed=42
        )
        audio = torch.randn(batch_size, 1000)

        # WHEN: Called with log=True
        audio_out, log_list = aug(audio, log=True)

        # THEN: Log should be a list with one entry per example
        assert isinstance(log_list, list)
        assert len(log_list) == batch_size
        # Each entry should be a dict
        for log_dict in log_list:
            assert isinstance(log_dict, dict)
            assert "augmentation" in log_dict
            assert "parameters" in log_dict

    def test_per_example_mode_logs_different_parameters_per_example(self, aug_cls):
        # GIVEN: An augmentation in per_example mode
        batch_size = 4
        aug = aug_cls(
            min_gain_db=-6.0, max_gain_db=6.0, p=1.0, mode="per_example", seed=42
        )
        audio = torch.randn(batch_size, 1000)

        # WHEN: Called with log=True
        audio_out, log_list = aug(audio, log=True)

        # THEN: Each example should have different parameters (with high probability)
        gains = [log["parameters"]["gain_db"] for log in log_list]
        # With 4 examples and a range of 12dB, very likely to have different values
        unique_gains = len(set(gains))
        assert unique_gains > 1, "Expected different gains per example"

    def test_per_example_mode_with_some_not_applied(self, aug_cls):
        # GIVEN: An augmentation with p=0.5 in per_example mode
        batch_size = 8
        # Note: GainAugmentation applies p check once for the whole batch, not per example
        # So we need to test this with a composition that can apply per-example
        # For now, let's test that the structure is correct when some DO apply
        aug = aug_cls(
            min_gain_db=-6.0, max_gain_db=6.0, p=1.0, mode="per_example", seed=42
        )
        audio = torch.randn(batch_size, 1000)

        # WHEN: Called with log=True
        audio_out, log_list = aug(audio, log=True)

        # THEN: Log list should have same length as batch
        assert len(log_list) == batch_size
        # All entries should be dicts (all applied since p=1.0)
        dict_count = sum(1 for log in log_list if isinstance(log, dict))
        assert dict_count == batch_size

    def test_per_example_mode_maintains_index_alignment_with_none(self, aug_cls):
        # GIVEN: An augmentation with p=0.5 in per_example mode (some will not apply)
        batch_size = 4
        aug = aug_cls(
            min_gain_db=-6.0, max_gain_db=6.0, p=0.5, mode="per_example", seed=123
        )
        audio = torch.randn(batch_size, 1000)

        # WHEN: Called with log=True
        audio_out, log_list = aug(audio, log=True)

        # THEN: Log list length should match batch size
        assert len(log_list) == batch_size
        # Each index should either have None or a dict
        for i, log in enumerate(log_list):
            assert log is None or isinstance(log, dict), (
                f"Index {i} has invalid log type: {type(log)}"
            )


@pytest.mark.parametrize(
    "comp_cls,expected_name",
    [(Compose, "Compose"), (OneOf, "OneOf"), (SomeOf, "SomeOf")],
)
class TestCompositionLogging:
    """Tests for logging in composition classes (Compose, OneOf, SomeOf)."""

    def test_composition_3d_input_returns_single_dict(self, comp_cls, expected_name):
        # GIVEN: A composition with 3D input (source, channel, time) - single example
        augmentations = {
            "gain": GainAugmentation(
                sample_rate=16000,
                min_gain_db=-6.0,
                max_gain_db=6.0,
                p=1.0,
                mode="per_batch",
                seed=42,
            ),
            "noise": NoiseAugmentation(
                sample_rate=16000,
                min_gain_db=-80,
                max_gain_db=-60,
                p=1.0,
                mode="per_batch",
                seed=43,
            ),
        }
        if comp_cls == SomeOf:
            aug = comp_cls(
                k=2, augmentations=augmentations, sample_rate=16000, mode="per_batch"
            )
        else:
            aug = comp_cls(augmentations, sample_rate=16000, mode="per_batch")
        # 3D tensor: (source, channel, time)
        audio = torch.randn(1, 1, 1000)

        # WHEN: Called with log=True
        audio_out, log_dict = aug(audio, log=True)

        # THEN: Log should be a single dict (not a list)
        assert isinstance(log_dict, dict), (
            f"Expected dict for 3D input, got {type(log_dict)}"
        )
        assert log_dict["augmentation"] == expected_name

    def test_composition_log_is_none_when_not_applied(self, comp_cls, expected_name):
        # GIVEN: A composition with p=0.0 and 3D input
        augmentations = {
            "gain": GainAugmentation(
                sample_rate=16000, min_gain_db=-6, max_gain_db=6, p=1.0, seed=42
            ),
        }
        if comp_cls == SomeOf:
            aug = comp_cls(
                k=1, augmentations=augmentations, sample_rate=16000, p=0.0, seed=100
            )
        else:
            aug = comp_cls(augmentations, sample_rate=16000, p=0.0)
        # 3D tensor: (source, channel, time)
        audio = torch.randn(1, 1, 1000)

        # WHEN: Called with log=True
        audio_out, log_dict = aug(audio, log=True)

        # THEN: Log should be None
        assert log_dict == [None]


class TestComposeLogging:
    """Tests for Compose-specific logging behavior."""

    def test_compose_with_log_returns_nested_dict(self):
        # GIVEN: A Compose with multiple augmentations
        aug = Compose(
            {
                "gain": GainAugmentation(
                    sample_rate=16000, min_gain_db=-6.0, max_gain_db=6.0, p=1.0, seed=42
                ),
                "noise": NoiseAugmentation(
                    sample_rate=16000, min_gain_db=-80, max_gain_db=-60, p=1.0, seed=43
                ),
            },
            sample_rate=16000,
            mode="per_batch",
        )
        audio = torch.randn(1, 1, 1000)

        # WHEN: Called with log=True
        audio_out, log_dict = aug(audio, log=True)

        # THEN: Log should be a dict with "augmentation" and "transforms" keys
        assert isinstance(log_dict, dict)
        assert log_dict["augmentation"] == "Compose"
        assert "transforms" in log_dict
        assert isinstance(log_dict["transforms"], dict)
        assert all([x in log_dict["transforms"] for x in ["gain", "noise"]])

    def test_compose_log_contains_all_applied_augmentations(self):
        # GIVEN: A Compose with multiple augmentations (all with p=1.0)
        aug = Compose(
            {
                "gain": GainAugmentation(
                    sample_rate=16000,
                    min_gain_db=-6,
                    max_gain_db=6,
                    p=1.0,
                    mode="per_batch",
                    seed=42,
                ),
                "noise": NoiseAugmentation(
                    sample_rate=16000,
                    min_gain_db=-80,
                    max_gain_db=-60,
                    p=1.0,
                    mode="per_batch",
                    seed=43,
                ),
            },
            sample_rate=16000,
            mode="per_batch",
        )
        audio = torch.randn(1, 1, 1000)

        # WHEN: Called with log=True
        audio_out, log_dict = aug(audio, log=True)

        # THEN: transforms dict should contain both augmentations
        transforms = log_dict["transforms"]
        assert "gain" in transforms
        assert "noise" in transforms
        assert transforms["gain"]["augmentation"] == "GainAugmentation"
        assert transforms["noise"]["augmentation"] == "NoiseAugmentation"

    def test_compose_log_omits_non_applied_augmentations(self):
        # GIVEN: A Compose where one augmentation has p=0 (never applies)
        aug = Compose(
            {
                "gain": GainAugmentation(
                    sample_rate=16000, min_gain_db=-6, max_gain_db=6, p=1.0, seed=42
                ),
                "noise": NoiseAugmentation(
                    sample_rate=16000, min_gain_db=-80, max_gain_db=-60, p=0.0, seed=43
                ),
            },
            sample_rate=16000,
            mode="per_batch",
        )
        audio = torch.randn(1, 1, 1000)

        # WHEN: Called with log=True
        audio_out, log_dict = aug(audio, log=True)

        # THEN: transforms dict should only contain the applied augmentation
        transforms = log_dict["transforms"]
        assert "gain" in transforms
        assert transforms["noise"] is None  # Omitted because not applied


class TestOneOfLogging:
    """Tests for OneOf-specific logging behavior."""

    def test_oneof_with_log_returns_dict_with_selected_augmentation(self):
        # GIVEN: A OneOf with multiple augmentations
        aug = OneOf(
            {
                "gain_high": GainAugmentation(
                    sample_rate=16000,
                    min_gain_db=3,
                    max_gain_db=6,
                    p=1.0,
                    mode="per_batch",
                    seed=42,
                ),
                "gain_low": GainAugmentation(
                    sample_rate=16000,
                    min_gain_db=-6,
                    max_gain_db=-3,
                    p=1.0,
                    mode="per_batch",
                    seed=43,
                ),
            },
            sample_rate=16000,
            seed=100,
            mode="per_batch",
        )
        audio = torch.randn(1, 1, 1000)

        # WHEN: Called with log=True
        audio_out, log_dict = aug(audio, log=True)

        # THEN: Log should contain augmentation name, selected key, and transform log
        assert isinstance(log_dict, dict)
        assert log_dict["augmentation"] == "OneOf"
        assert len(log_dict["parameters"]) == 1
        assert "transforms" in log_dict
        # Selected should be one of the keys
        assert log_dict["parameters"][0] in ["gain_high", "gain_low"]
        # Transform should be the log from the selected augmentation
        assert isinstance(log_dict["transforms"], dict)
        transform_detail = log_dict["transforms"][log_dict["parameters"][0]]
        assert transform_detail["augmentation"] == "GainAugmentation"


class TestSomeOfLogging:
    """Tests for SomeOf-specific logging behavior."""

    def test_someof_with_log_returns_dict_with_selected_augmentations(self):
        # GIVEN: A SomeOf with k=2
        aug = SomeOf(
            k=2,
            augmentations={
                "gain": GainAugmentation(
                    sample_rate=16000, min_gain_db=-6, max_gain_db=6, p=1.0, seed=42
                ),
                "noise": NoiseAugmentation(
                    sample_rate=16000, min_gain_db=-80, max_gain_db=-60, p=1.0, seed=43
                ),
                "gain2": GainAugmentation(
                    sample_rate=16000, min_gain_db=-3, max_gain_db=3, p=1.0, seed=44
                ),
            },
            sample_rate=16000,
            seed=100,
            mode="per_batch",
        )
        audio = torch.randn(1, 1, 1000)

        # WHEN: Called with log=True
        audio_out, log_dict = aug(audio, log=True)

        # THEN: Log should contain augmentation name, selected keys, and transforms
        assert isinstance(log_dict, dict)
        assert log_dict["augmentation"] == "SomeOf"
        assert len(log_dict["parameters"]) == 2

        # Transforms should be a dict with the selected augmentations
        assert isinstance(log_dict["transforms"], dict)
        assert len(log_dict["transforms"]) == 2
        # Each selected key should be in transforms
        for key in log_dict["parameters"]:
            assert key in log_dict["transforms"]

    def test_someof_with_k_equals_zero_returns_empty_log(self):
        # GIVEN: A SomeOf with k=0
        aug = SomeOf(
            k=0,
            augmentations={
                "gain": GainAugmentation(
                    sample_rate=16000, min_gain_db=-6, max_gain_db=6, p=1.0, seed=42
                ),
            },
            sample_rate=16000,
            seed=100,
            mode="per_batch",
        )
        audio = torch.randn(1, 1, 1000)

        # WHEN: Called with log=True
        audio_out, log_dict = aug(audio, log=True)

        # THEN: Log should have empty selected and transforms
        assert log_dict["augmentation"] == "SomeOf"
        assert len(log_dict["parameters"]) == 0
        assert log_dict["transforms"] == {}


@pytest.mark.parametrize(
    "comp_cls,expected_name",
    [(Compose, "Compose"), (OneOf, "OneOf"), (SomeOf, "SomeOf")],
)
class TestCompositionPerBatchLogging:
    """Tests for logging in composition classes with explicit per_batch mode."""

    def test_composition_4d_input_per_batch_mode_returns_single_dict(
        self, comp_cls, expected_name
    ):
        # GIVEN: A composition with EXPLICIT mode="per_batch" and 4D input
        # This overrides the default per_example behavior for 4D input
        batch_size = 3
        augmentations = {
            "gain": GainAugmentation(
                sample_rate=16000,
                min_gain_db=-6,
                max_gain_db=6,
                p=1.0,
                mode="per_batch",
                seed=42,
            ),
        }
        if comp_cls == SomeOf:
            aug = comp_cls(
                k=1, augmentations=augmentations, sample_rate=16000, mode="per_batch"
            )
        else:
            aug = comp_cls(augmentations, sample_rate=16000, mode="per_batch")
        # 4D tensor: (batch, source, channel, time)
        audio = torch.randn(batch_size, 1, 1, 1000)

        # WHEN: Called with log=True
        audio_out, log_dict = aug(audio, log=True)

        # THEN: Log should be a single dict (not a list) because mode="per_batch"
        assert isinstance(log_dict, dict), (
            f"Expected dict for per_batch mode, got {type(log_dict)}"
        )
        assert log_dict["augmentation"] == expected_name


@pytest.mark.parametrize(
    "comp_cls,expected_name",
    [(Compose, "Compose"), (OneOf, "OneOf"), (SomeOf, "SomeOf")],
)
class TestCompositionPerExampleLogging:
    """Tests for logging in composition classes with per_example mode (4D input)."""

    def test_composition_4d_input_defaults_to_per_example_returns_list(
        self, comp_cls, expected_name
    ):
        # GIVEN: A composition with 4D input (batch, source, channel, time)
        # WITHOUT explicitly setting mode (should default to per_example)
        batch_size = 4
        augmentations = {
            "gain": GainAugmentation(
                sample_rate=16000,
                min_gain_db=-6,
                max_gain_db=6,
                p=1.0,
                mode="per_batch",
                seed=42,
            ),
            "noise": NoiseAugmentation(
                sample_rate=16000,
                min_gain_db=-80,
                max_gain_db=-60,
                p=1.0,
                mode="per_batch",
                seed=43,
            ),
        }
        if comp_cls == SomeOf:
            aug = comp_cls(k=2, augmentations=augmentations, sample_rate=16000)
        else:
            aug = comp_cls(augmentations, sample_rate=16000)

        # Shape: (batch, source, channel, time) - 4D
        audio = torch.randn(batch_size, 1, 1, 1000)

        # WHEN: Called with log=True
        audio_out, log_list = aug(audio, log=True)

        # THEN: Log should be a list (defaults to per_example for 4D input)
        assert isinstance(log_list, list), (
            f"Expected list for 4D input (default per_example), got {type(log_list)}"
        )
        assert len(log_list) == batch_size, (
            f"Expected {batch_size} logs, got {len(log_list)}"
        )
        # Each entry should be a dict
        for i, log_dict in enumerate(log_list):
            assert isinstance(log_dict, dict), (
                f"Example {i}: expected dict, got {type(log_dict)}"
            )
            assert log_dict["augmentation"] == expected_name

    def test_composition_per_example_explicit_mode_returns_list_of_logs(
        self, comp_cls, expected_name
    ):
        # GIVEN: A composition with explicit mode="per_example"
        batch_size = 4
        augmentations = {
            "gain": GainAugmentation(
                sample_rate=16000,
                min_gain_db=-6,
                max_gain_db=6,
                p=1.0,
                mode="per_batch",
                seed=42,
            ),
            "noise": NoiseAugmentation(
                sample_rate=16000,
                min_gain_db=-80,
                max_gain_db=-60,
                p=1.0,
                mode="per_batch",
                seed=43,
            ),
        }
        if comp_cls == SomeOf:
            aug = comp_cls(
                k=2, augmentations=augmentations, sample_rate=16000, mode="per_example"
            )
        else:
            aug = comp_cls(augmentations, sample_rate=16000, mode="per_example")

        # Shape: (batch, source, channel, time) for compositions
        audio = torch.randn(batch_size, 1, 1, 1000)

        # WHEN: Called with log=True
        audio_out, log_list = aug(audio, log=True)

        # THEN: Log should be a list with one entry per example
        assert isinstance(log_list, list), (
            f"Expected list for per_example mode, got {type(log_list)}"
        )
        assert len(log_list) == batch_size, (
            f"Expected {batch_size} logs, got {len(log_list)}"
        )
        # Each entry should be a dict
        for i, log_dict in enumerate(log_list):
            assert isinstance(log_dict, dict), (
                f"Example {i}: expected dict, got {type(log_dict)}"
            )
            assert log_dict["augmentation"] == expected_name

    def test_composition_per_example_logs_have_correct_structure(
        self, comp_cls, expected_name
    ):
        # GIVEN: A composition in per_example mode
        batch_size = 3
        augmentations = {
            "gain": GainAugmentation(
                sample_rate=16000,
                min_gain_db=-6,
                max_gain_db=6,
                p=1.0,
                mode="per_batch",
                seed=42,
            ),
        }
        if comp_cls == SomeOf:
            aug = comp_cls(
                k=1, augmentations=augmentations, sample_rate=16000, mode="per_example"
            )
        else:
            aug = comp_cls(augmentations, sample_rate=16000, mode="per_example")

        audio = torch.randn(batch_size, 1, 1, 1000)

        # WHEN: Called with log=True
        audio_out, log_list = aug(audio, log=True)

        # THEN: Each log should have the expected structure
        for i, log_dict in enumerate(log_list):
            assert "augmentation" in log_dict, (
                f"Example {i}: missing 'augmentation' key"
            )
            if comp_cls == Compose:
                assert "transforms" in log_dict, (
                    f"Example {i}: missing 'transforms' key"
                )
                assert isinstance(log_dict["transforms"], dict)
            elif comp_cls == OneOf:
                assert "parameters" in log_dict, (
                    f"Example {i}: missing 'parameters' key"
                )
                assert "transforms" in log_dict, (
                    f"Example {i}: missing 'transforms' key"
                )
            elif comp_cls == SomeOf:
                assert "parameters" in log_dict, (
                    f"Example {i}: missing 'parameters' key"
                )
                assert "transforms" in log_dict, (
                    f"Example {i}: missing 'transforms' key"
                )

    def test_composition_per_example_different_logs_per_example(
        self, comp_cls, expected_name
    ):
        # GIVEN: A composition in per_example mode with range parameters
        batch_size = 4
        augmentations = {
            "gain": GainAugmentation(
                sample_rate=16000,
                min_gain_db=-6,
                max_gain_db=6,
                p=1.0,
                mode="per_batch",
                seed=42,
            ),
        }
        if comp_cls == SomeOf:
            aug = comp_cls(
                k=1, augmentations=augmentations, sample_rate=16000, mode="per_example"
            )
        else:
            aug = comp_cls(augmentations, sample_rate=16000, mode="per_example")

        audio = torch.randn(batch_size, 1, 1, 1000)

        # WHEN: Called with log=True
        audio_out, log_list = aug(audio, log=True)

        # THEN: Different examples should have different parameter values (with high probability)
        # Extract the gain_db values from each example's log
        gain_values = []
        for log_dict in log_list:
            gain_db = log_dict["transforms"]["gain"]["parameters"]["gain_db"]
            gain_values.append(gain_db)

        # With 4 examples and a range of 12dB, very likely to have different values
        unique_gains = len(set(gain_values))
        assert unique_gains > 1, (
            f"Expected different gains per example in per_example mode, "
            f"got all same: {gain_values}"
        )


class TestPeakFilterCompositionLogging:
    """Tests for logging with PeakFilter in compositions."""

    def test_peak_filter_in_someof_with_logging(self):
        """GIVEN PeakFilter in SomeOf, WHEN called with log=True, THEN should return log."""
        # GIVEN: A SomeOf composition containing PeakFilter

        aug = SomeOf(
            k=1,
            augmentations={
                "peak": PeakFilter(
                    sample_rate=16000,
                    min_center_freq=1000.0,
                    max_center_freq=1000.0,
                    min_gain_db=6.0,
                    max_gain_db=6.0,
                    p=1.0,
                ),
            },
            sample_rate=16000,
            p=1.0,
        )
        audio = torch.randn(2, 2, 1, 16000) * 0.5

        # WHEN: Called with log=True
        audio_out, log_dicts = aug(audio, log=True)

        # THEN: Should return log dict with PeakFilter info
        for log_dict in log_dicts:
            assert isinstance(log_dict, dict)
            assert log_dict["augmentation"] == "SomeOf"
            assert "parameters" in log_dict
            assert "peak" in log_dict["parameters"]
            assert "transforms" in log_dict
            assert "peak" in log_dict["transforms"]
            peak_logs = log_dict["transforms"]["peak"]
            for peak_log in peak_logs:
                assert peak_log["augmentation"] == "PeakFilter"
                assert "parameters" in peak_log

    def test_peak_filter_in_oneof_with_logging(self):
        """GIVEN PeakFilter in OneOf, WHEN called with log=True, THEN should return log."""
        # GIVEN: A OneOf composition containing PeakFilter

        aug = OneOf(
            augmentations={
                "peak": PeakFilter(
                    sample_rate=16000,
                    min_center_freq=1000.0,
                    max_center_freq=1000.0,
                    min_gain_db=6.0,
                    max_gain_db=6.0,
                    p=1.0,
                ),
            },
            sample_rate=16000,
            p=1.0,
        )
        audio = torch.randn(2, 2, 1, 16000) * 0.5

        # WHEN: Called with log=True
        audio_out, log_dicts = aug(audio, log=True)

        # THEN: Should return log dict with PeakFilter info
        for log_dict in log_dicts:
            assert isinstance(log_dict, dict)
            assert log_dict["augmentation"] == "OneOf"
            assert "parameters" in log_dict
            assert log_dict["parameters"] == ["peak"]
            assert "transforms" in log_dict
            peak_logs = log_dict["transforms"]["peak"]
            for peak_log in peak_logs:
                assert peak_log["augmentation"] == "PeakFilter"
                assert "parameters" in peak_log
