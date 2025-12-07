"""Tests for db/amplitude conversion utilities."""

import torch
import pytest

from auementations.utils import (
    db_to_power,
    db_to_amplitude,
    power_to_db,
    amplitude_to_db,
)


class TestDbToPower:
    """Tests for db_to_power function."""

    def test_converts_zero_db_to_reference_power(self):
        # GIVEN: 0 dB input with default reference
        S_db = torch.tensor([0.0])

        # WHEN: Converting to power
        result = db_to_power(S_db)

        # THEN: Output equals reference power (1.0)
        assert torch.allclose(result, torch.tensor([1.0]))

    def test_converts_positive_db_to_higher_power(self):
        # GIVEN: 10 dB input
        S_db = torch.tensor([10.0])

        # WHEN: Converting to power
        result = db_to_power(S_db)

        # THEN: Output is 10x reference (10^(10/10) = 10)
        assert torch.allclose(result, torch.tensor([10.0]))

    def test_converts_negative_db_to_lower_power(self):
        # GIVEN: -10 dB input
        S_db = torch.tensor([-10.0])

        # WHEN: Converting to power
        result = db_to_power(S_db)

        # THEN: Output is 0.1x reference (10^(-10/10) = 0.1)
        assert torch.allclose(result, torch.tensor([0.1]))

    def test_applies_custom_reference(self):
        # GIVEN: 0 dB with custom reference
        S_db = torch.tensor([0.0])
        ref = 2.0

        # WHEN: Converting to power
        result = db_to_power(S_db, ref=ref)

        # THEN: Output equals custom reference
        assert torch.allclose(result, torch.tensor([2.0]))

    def test_handles_multidimensional_tensors(self):
        # GIVEN: Multi-dimensional dB tensor
        S_db = torch.tensor([[0.0, 10.0], [-10.0, 20.0]])

        # WHEN: Converting to power
        result = db_to_power(S_db)

        # THEN: All values converted correctly
        expected = torch.tensor([[1.0, 10.0], [0.1, 100.0]])
        assert torch.allclose(result, expected)


class TestDbToAmplitude:
    """Tests for db_to_amplitude function."""

    def test_converts_zero_db_to_reference_amplitude(self):
        # GIVEN: 0 dB input with default reference
        S_db = torch.tensor([0.0])

        # WHEN: Converting to amplitude
        result = db_to_amplitude(S_db)

        # THEN: Output equals reference amplitude (1.0)
        assert torch.allclose(result, torch.tensor([1.0]))

    def test_converts_positive_db_to_higher_amplitude(self):
        # GIVEN: 20 dB input (doubles amplitude in power domain)
        S_db = torch.tensor([20.0])

        # WHEN: Converting to amplitude
        result = db_to_amplitude(S_db)

        # THEN: Output is 10x reference (sqrt(10^(20/10)) = 10)
        assert torch.allclose(result, torch.tensor([10.0]))

    def test_converts_negative_db_to_lower_amplitude(self):
        # GIVEN: -20 dB input
        S_db = torch.tensor([-20.0])

        # WHEN: Converting to amplitude
        result = db_to_amplitude(S_db)

        # THEN: Output is 0.1x reference
        assert torch.allclose(result, torch.tensor([0.1]))

    def test_applies_custom_reference(self):
        # GIVEN: 0 dB with custom reference
        S_db = torch.tensor([0.0])
        ref = 3.0

        # WHEN: Converting to amplitude
        result = db_to_amplitude(S_db, ref=ref)

        # THEN: Output equals custom reference
        assert torch.allclose(result, torch.tensor([3.0]))


class TestPowerToDb:
    """Tests for power_to_db function."""

    def test_converts_reference_power_to_zero_db(self):
        # GIVEN: Power equal to reference
        S = torch.tensor([1.0])

        # WHEN: Converting to dB
        result = power_to_db(S)

        # THEN: Output is 0 dB
        assert torch.allclose(result, torch.tensor([0.0]))

    def test_converts_higher_power_to_positive_db(self):
        # GIVEN: Power 10x reference
        S = torch.tensor([10.0])

        # WHEN: Converting to dB
        result = power_to_db(S)

        # THEN: Output is 10 dB
        assert torch.allclose(result, torch.tensor([10.0]))

    def test_converts_lower_power_to_negative_db(self):
        # GIVEN: Power 0.1x reference
        S = torch.tensor([0.1])

        # WHEN: Converting to dB
        result = power_to_db(S)

        # THEN: Output is -10 dB
        assert torch.allclose(result, torch.tensor([-10.0]))

    def test_applies_amin_threshold(self):
        # GIVEN: Very small power values
        S = torch.tensor([1e-20])
        amin = 1e-10

        # WHEN: Converting to dB with amin
        result = power_to_db(S, amin=amin)

        # THEN: Values clamped to amin
        expected = power_to_db(torch.tensor([amin]), amin=amin)
        assert torch.allclose(result, expected)

    def test_applies_top_db_threshold(self):
        # GIVEN: Wide dynamic range
        S = torch.tensor([1.0, 1e-10])
        top_db = 40.0

        # WHEN: Converting to dB with top_db
        result = power_to_db(S, top_db=top_db)

        # THEN: Range limited to top_db below peak
        dynamic_range = result.max() - result.min()
        assert dynamic_range <= top_db + 0.01  # Small tolerance

    def test_uses_callable_reference(self):
        # GIVEN: Power tensor and max reference function
        S = torch.tensor([1.0, 10.0, 100.0])

        # WHEN: Converting with torch.max as reference
        result = power_to_db(S, ref=torch.max)

        # THEN: Maximum value is 0 dB
        assert torch.allclose(result.max(), torch.tensor([0.0]))

    def test_handles_complex_input_with_warning(self):
        # GIVEN: Complex-valued tensor
        S = torch.tensor([1.0 + 1.0j, 2.0 + 2.0j])

        # WHEN: Converting to dB
        with pytest.warns(UserWarning, match="phase information will be discarded"):
            result = power_to_db(S)

        # THEN: Magnitude is used
        expected = power_to_db(torch.abs(S))
        assert torch.allclose(result, expected)

    def test_raises_on_non_positive_amin(self):
        # GIVEN: Invalid amin value
        S = torch.tensor([1.0])

        # WHEN/THEN: Raises ValueError
        with pytest.raises(ValueError, match="amin must be strictly positive"):
            power_to_db(S, amin=0.0)

    def test_raises_on_negative_top_db(self):
        # GIVEN: Invalid top_db value
        S = torch.tensor([1.0])

        # WHEN/THEN: Raises ValueError
        with pytest.raises(ValueError, match="top_db must be non-negative"):
            power_to_db(S, top_db=-10.0)


class TestAmplitudeToDb:
    """Tests for amplitude_to_db function."""

    def test_converts_reference_amplitude_to_zero_db(self):
        # GIVEN: Amplitude equal to reference
        S = torch.tensor([1.0])

        # WHEN: Converting to dB
        result = amplitude_to_db(S)

        # THEN: Output is 0 dB
        assert torch.allclose(result, torch.tensor([0.0]))

    def test_converts_higher_amplitude_to_positive_db(self):
        # GIVEN: Amplitude 10x reference
        S = torch.tensor([10.0])

        # WHEN: Converting to dB
        result = amplitude_to_db(S)

        # THEN: Output is 20 dB (20*log10(10) = 20)
        assert torch.allclose(result, torch.tensor([20.0]))

    def test_converts_lower_amplitude_to_negative_db(self):
        # GIVEN: Amplitude 0.1x reference
        S = torch.tensor([0.1])

        # WHEN: Converting to dB
        result = amplitude_to_db(S)

        # THEN: Output is -20 dB
        assert torch.allclose(result, torch.tensor([-20.0]))

    def test_uses_callable_reference(self):
        # GIVEN: Amplitude tensor and max reference function
        S = torch.tensor([1.0, 5.0, 10.0])

        # WHEN: Converting with torch.max as reference
        result = amplitude_to_db(S, ref=torch.max)

        # THEN: Maximum value is 0 dB
        assert torch.allclose(result.max(), torch.tensor([0.0]))

    def test_handles_complex_input_with_warning(self):
        # GIVEN: Complex-valued tensor
        S = torch.tensor([1.0 + 0.0j, 2.0 + 0.0j])

        # WHEN: Converting to dB
        with pytest.warns(UserWarning, match="phase information will be discarded"):
            result = amplitude_to_db(S)

        # THEN: Magnitude is used
        expected = amplitude_to_db(torch.abs(S))
        assert torch.allclose(result, expected)


class TestRoundtripConversions:
    """Tests for roundtrip conversions (db->linear->db and linear->db->linear)."""

    def test_power_roundtrip(self):
        # GIVEN: Original power values
        original = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0])

        # WHEN: Converting to dB and back
        db = power_to_db(original)
        reconstructed = db_to_power(db)

        # THEN: Values are preserved
        assert torch.allclose(reconstructed, original, rtol=1e-5)

    def test_amplitude_roundtrip(self):
        # GIVEN: Original amplitude values
        original = torch.tensor([0.1, 0.5, 1.0, 5.0, 10.0])

        # WHEN: Converting to dB and back
        db = amplitude_to_db(original)
        reconstructed = db_to_amplitude(db)

        # THEN: Values are preserved
        assert torch.allclose(reconstructed, original, rtol=1e-5)

    def test_db_to_power_to_db_roundtrip(self):
        # GIVEN: Original dB values
        original = torch.tensor([-40.0, -20.0, 0.0, 20.0, 40.0])

        # WHEN: Converting to power and back
        power = db_to_power(original)
        reconstructed = power_to_db(power)

        # THEN: Values are preserved
        assert torch.allclose(reconstructed, original, rtol=1e-5)

    def test_db_to_amplitude_to_db_roundtrip(self):
        # GIVEN: Original dB values
        original = torch.tensor([-40.0, -20.0, 0.0, 20.0, 40.0])

        # WHEN: Converting to amplitude and back
        amplitude = db_to_amplitude(original)
        reconstructed = amplitude_to_db(amplitude)

        # THEN: Values are preserved
        assert torch.allclose(reconstructed, original, rtol=1e-5)

    def test_roundtrip_with_custom_reference(self):
        # GIVEN: Original values and custom reference
        original = torch.tensor([1.0, 2.0, 3.0])
        ref = 2.5

        # WHEN: Converting amplitude to dB and back with same reference
        db = amplitude_to_db(original, ref=ref)
        reconstructed = db_to_amplitude(db, ref=ref)

        # THEN: Values are preserved
        assert torch.allclose(reconstructed, original, rtol=1e-5)
