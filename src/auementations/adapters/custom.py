from typing import Any

import torch
from torch import Tensor

from auementations.config.config_store import auementations_store
from auementations.core.base import BaseAugmentation
from auementations.utils import amplitude_to_db, db_to_amplitude

__all__ = ["GainAugmentation", "NoiseAugmentation", "NormAugmentation"]


@auementations_store(name="gain", group="auementations")
class GainAugmentation(BaseAugmentation):
    VALID_MODES = ["per_batch", "per_example", "per_source", "per_channel"]

    def __init__(
        self,
        sample_rate: int | float | None = None,
        min_gain_db: int | float = -12.0,
        max_gain_db: int | float = 12.0,
        p: float = 1.0,
        mode: str = "per_example",
        seed: int | None = None,
    ):
        super().__init__(sample_rate=sample_rate, p=p, mode=mode, seed=seed)
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        # For logging
        self._last_gains_db = None

    def randomize_parameters(self) -> dict[str, Any]:
        """Compatibility method for compositions. GainAugmentation does randomization internally."""
        uniform_0_1_val = self.rng.random()

        random_gain_in_db = (
            self.max_gain_db - self.min_gain_db
        ) * uniform_0_1_val + self.min_gain_db

        self._last_gains_db = random_gain_in_db

        return {
            "gain_db": self._last_gains_db.item()
            if isinstance(self._last_gains_db, torch.Tensor)
            else self._last_gains_db
        }

    def forward(self, audio: Tensor) -> Tensor:
        signal_max = audio.abs().amax().flatten()
        signal_max_db = amplitude_to_db(signal_max, amin=1e-8, top_db=100.0)
        max_gain_db_to_not_clip = min(self._last_gains_db, signal_max_db.item() * -1)

        random_gain_in_a = db_to_amplitude(torch.as_tensor(max_gain_db_to_not_clip))

        audio_ = audio * random_gain_in_a
        return audio_


@auementations_store(name="noise", group="auementations")
class NoiseAugmentation(BaseAugmentation):
    """Apply a constant gain noise to each example in a batch."""

    NOISE_MODES = ["fixed_db", "fixed_amp", "range_db"]

    def __init__(
        self,
        sample_rate: int | float | None = None,
        p: float = 1.0,
        min_gain_db: float = -100.0,
        max_gain_db: float = -75.0,
        gain_db: float | None = None,
        gain_amp: float | None = None,
        mode: str = "per_batch",
        seed: int | None = None,
    ):
        super().__init__(sample_rate=sample_rate, p=p, mode=mode, seed=seed)
        self.min_gain_db: float | None = None
        self.max_gain_db: float | None = None
        self.gain_db: float | None = None
        self.gain_amp: float | None = None

        if gain_db is not None:
            self.noise_mode = "fixed_db"
            self.gain_db = gain_db

        elif gain_amp is not None:
            self.noise_mode = "fixed_amp"
            self.gain_amp = gain_amp

        else:
            self.noise_mode = "range_db"
            self.min_gain_db = min_gain_db
            self.max_gain_db = max_gain_db

        # For logging
        self._last_gain_db = None

        self.torch_gen = torch.Generator()
        if seed is not None:
            self.torch_gen.manual_seed(seed)

    def randomize_parameters(self) -> dict[str, Any]:
        """Compatibility method for compositions. NoiseAugmentation does randomization internally."""
        gain_db = 0.0

        match self.noise_mode:
            case "fixed_db":
                gain_db = torch.as_tensor(self.gain_db)

            case "fixed_amp":
                gain_db = amplitude_to_db(
                    torch.as_tensor(self.gain_amp), amin=1e-8, top_db=100.0
                )

            case "range_db":
                uniform_0_1_val = self.rng.random()

                gain_db = (
                    self.max_gain_db - self.min_gain_db
                ) * uniform_0_1_val + self.min_gain_db

        self._last_gain_db = gain_db

        return {
            "gain_db": self._last_gain_db.item()
            if isinstance(self._last_gain_db, torch.Tensor)
            else self._last_gain_db
        }

    def forward(self, x):
        noise_sig = torch.empty_like(x).uniform_(generator=self.torch_gen) * 2 - 1

        random_gain_in_a = db_to_amplitude(self._last_gain_db)
        noise_sig = noise_sig * random_gain_in_a

        return x + noise_sig


@auementations_store(name="norm", group="auementations")
class NormAugmentation(BaseAugmentation):
    """Apply a normalization to the signal.

    You can apply it as a randon range of norm values by passing a tuple.
    You can also *only* normalize if the signal is above a threshold by passing
     a threshold.
    """

    def __init__(
        self,
        sample_rate: int | float | None = None,
        p: float = 1.0,
        mode: str = "per_batch",
        seed: int | None = None,
        set_norm_to_dbfs: float | tuple[float] = 1.0,
        threshold_dbfs: float | None = None,
    ):
        super().__init__(sample_rate=sample_rate, p=p, mode=mode, seed=seed)

        self.set_norm_to_dbfs = set_norm_to_dbfs
        self.threshold = (
            db_to_amplitude(torch.as_tensor(threshold_dbfs))
            if threshold_dbfs is not None
            else None
        )

    def randomize_parameters(self) -> dict[str, Any]:
        if isinstance(self.set_norm_to_dbfs, float):
            self.selected_max_db = self.set_norm_to_dbfs
        elif (
            isinstance(self.set_norm_to_dbfs, (tuple, Tensor))
            and len(self.set_norm_to_dbfs) == 2
        ):
            self.selected_max_db = self.rng.uniform(*self.set_norm_to_dbfs)
        else:
            self.selected_max_db = 0.0

        self.selected_max_a = db_to_amplitude(torch.as_tensor(self.selected_max_db))

        return {"norm_value": self.selected_max_a, "threshold": self.threshold}

    def forward(self, x):
        x_max = x.amax()

        if self.threshold is None or (
            self.threshold is not None and x_max > self.threshold
        ):
            scale_value = self.selected_max_a / x_max

        else:  # self.threshold is not None:
            scale_value = 1.0

        return x * scale_value


# @auementations_store(name="relative_noise", group="auementations")
# class RelativeNoiseAugmentation(nn.Module):
#     """Apply a noise a specified db less than the signal."""

#     VALID_MODES = ["per_example"]

#     def __init__(self, sample_rate: int | float | None = None):
#         self.generator = torch.Generator()
#         if self.seed is not None:
#             self.generator.manual_seed(self.seed)

#     def noise_from_mode(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.rand_like(x) * 2 - 1

#     def forward(self, x):
#         p_apply = torch.rand((), generator=self.generator)

#         if p_apply <= self.p:
#             noise = self.noise_from_mode(x)

#             return x + noise
