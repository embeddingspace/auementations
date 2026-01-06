from typing import Any

import torch
from torch import Tensor

from auementations.config.config_store import auementations_store
from auementations.core.base import BaseAugmentation
from auementations.utils import amplitude_to_db, db_to_amplitude

__all__ = ["GainAugmentation", "NoiseAugmentation"]


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
        # self._last_applied = False

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

    # def __call__(self, x: torch.Tensor, log: bool = False):
    #     """Apply augmentation with optional logging.

    #     Args:
    #         x: Input audio tensor.
    #         log: If True, return (audio, log_dict). If False, return audio only.

    #     Returns:
    #         If log=False: Augmented audio tensor.
    #         If log=True: Tuple of (augmented_audio, log_dict).
    #     """
    #     # Reset logging state
    #     self._last_gains_db = None
    #     self._last_applied = False

    #     # Call forward
    #     output = self.forward(x)

    #     if not log:
    #         return output

    #     # Build log dict
    #     if not self._last_applied:
    #         log_dict = None
    #     elif self.mode == "per_example" and isinstance(self._last_gains_db, list):
    #         # Per-example mode: return list of dicts
    #         log_dict = []
    #         for gain_db in self._last_gains_db:
    #             if gain_db is not None:
    #                 log_dict.append(
    #                     {
    #                         "augmentation": self.__class__.__name__,
    #                         "parameters": {"gain_db": gain_db},
    #                     }
    #                 )
    #             else:
    #                 log_dict.append(None)
    #     else:
    #         # Single gain or per_batch mode
    #         log_dict = {
    #             "augmentation": self.__class__.__name__,
    #             "parameters": {"gain_db": self._last_gains_db},
    #         }

    #     return output, log_dict

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     p_apply = torch.rand((), generator=self.generator)

    #     if p_apply <= self.p:
    #         self._last_applied = True
    #         # Determine the reduction dimensions and gain shape based on mode
    #         match self.mode:
    #             case "per_example":
    #                 # One gain per example (batch dimension)
    #                 num_gains = x.shape[0]
    #                 gain_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    #                 # Reduce over all dimensions except batch
    #                 dims_to_reduce = tuple(range(1, x.ndim))
    #                 signal_max = x.abs().amax(dim=dims_to_reduce, keepdim=False)
    #             case "per_source":
    #                 # One gain per source (dimension 1)
    #                 if x.ndim < 2:
    #                     raise ValueError(
    #                         f"per_source mode requires at least 2 dimensions, got {x.ndim}"
    #                     )
    #                 num_gains = x.shape[0] * x.shape[1]
    #                 gain_shape = (x.shape[0], x.shape[1]) + (1,) * (x.ndim - 2)
    #                 # Reduce over all dimensions except batch and source
    #                 dims_to_reduce = tuple(range(2, x.ndim))
    #                 signal_max = (
    #                     x.abs().amax(dim=dims_to_reduce, keepdim=False).flatten()
    #                 )
    #             case "per_channel":
    #                 # One gain per channel (dimension 2)
    #                 if x.ndim < 3:
    #                     raise ValueError(
    #                         f"per_channel mode requires at least 3 dimensions, got {x.ndim}"
    #                     )
    #                 num_gains = x.shape[0] * x.shape[1] * x.shape[2]
    #                 gain_shape = (x.shape[0], x.shape[1], x.shape[2]) + (1,) * (
    #                     x.ndim - 3
    #                 )
    #                 # Reduce over all dimensions except batch, source, and channel
    #                 dims_to_reduce = tuple(range(3, x.ndim))
    #                 signal_max = (
    #                     x.abs().amax(dim=dims_to_reduce, keepdim=False).flatten()
    #                 )
    #             # case "per_batch":
    #             case _:
    #                 # Single gain for entire batch
    #                 num_gains = 1
    #                 gain_shape = (1,) * x.ndim  # Will broadcast to all dimensions
    #                 # Reduce over all dimensions
    #                 signal_max = x.abs().max()

    #         # Sample random gain values
    #         uniform_0_1_tensor = torch.rand((num_gains,), generator=self.generator)

    #         # Prevent gain from going > 1.0 by clipping the gain max
    #         # Convert signal max to dB and compute max allowable gain
    #         signal_max_db = amplitude_to_db(signal_max)

    #         if self.mode == "per_batch":
    #             # Scalar case - keep as tensor
    #             max_gain_db_scalar = min(self.max_gain_db, signal_max_db.item() * -1)
    #             max_gain_db = torch.tensor(
    #                 max_gain_db_scalar,
    #                 dtype=signal_max_db.dtype,
    #                 device=signal_max_db.device,
    #             )
    #         else:
    #             # Vectorized version for multiple gains
    #             max_gain_db = torch.minimum(
    #                 torch.tensor(
    #                     self.max_gain_db,
    #                     dtype=signal_max_db.dtype,
    #                     device=signal_max_db.device,
    #                 ),
    #                 signal_max_db * -1,
    #             )

    #         # Calculate random gain in dB (works for both scalar and vector cases)

    #         # Store for logging
    #         if self.mode == "per_example":
    #             # Convert tensor to list of floats for per-example logging
    #             self._last_gains_db = random_gain_in_db.flatten().tolist()
    #         elif self.mode == "per_batch":
    #             # Store as single scalar
    #             self._last_gains_db = random_gain_in_db.item()
    #         else:
    #             # For per_source and per_channel modes, store the flattened list
    #             self._last_gains_db = random_gain_in_db.flatten().tolist()

    #         # Convert to amplitude and reshape
    #         random_gain_in_a = db_to_amplitude(random_gain_in_db)
    #         random_gain_in_a = random_gain_in_a.reshape(gain_shape)

    #         # Apply gain
    #         x = x * random_gain_in_a

    #     return x

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

    # def __call__(self, x: torch.Tensor, log: bool = False):
    #     """Apply augmentation with optional logging.

    #     Args:
    #         x: Input audio tensor.
    #         log: If True, return (audio, log_dict). If False, return audio only.

    #     Returns:
    #         If log=False: Augmented audio tensor.
    #         If log=True: Tuple of (augmented_audio, log_dict).
    #     """
    #     # Reset logging state
    #     self._last_gain_db = None
    #     self._last_applied = False

    #     # Call forward
    #     output = self.forward(x)

    #     if not log:
    #         return output

    #     # Build log dict
    #     if not self._last_applied:
    #         log_dict = None
    #     elif self.mode == "per_example" and isinstance(self._last_gain_db, list):
    #         # Per-example mode: return list of dicts
    #         log_dict = []
    #         for gain_db in self._last_gain_db:
    #             if gain_db is not None:
    #                 log_dict.append(
    #                     {
    #                         "augmentation": self.__class__.__name__,
    #                         "parameters": {"gain_db": gain_db},
    #                     }
    #                 )
    #             else:
    #                 log_dict.append(None)
    #     else:
    #         # Single gain or per_batch mode
    #         log_dict = {
    #             "augmentation": self.__class__.__name__,
    #             "parameters": {"gain_db": self._last_gain_db},
    #         }

    #     return output, log_dict

    # def noise_from_mode(self, x: torch.Tensor) -> torch.Tensor:

    #     match self.noise_mode:
    #         case "fixed_db":
    #             random_gain_in_a = db_to_amplitude(torch.as_tensor(self.gain_db))
    #             # Store for logging
    #             self._last_gain_db = self.gain_db

    #         case "fixed_amp":
    #             random_gain_in_a = self.gain_amp
    #             # Convert to dB for logging
    #             self._last_gain_db = amplitude_to_db(
    #                 torch.as_tensor(self.gain_amp)
    #             ).item()

    #         case "range_db":
    #             match self.mode:
    #                 case "per_batch":
    #                     uniform_0_1_tensor = torch.rand(1, generator=self.generator)
    #                 case "per_example":
    #                     num_gains = x.shape[0]
    #                     gain_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    #                     uniform_0_1_tensor = torch.rand(
    #                         (num_gains,), generator=self.generator
    #                     )
    #                     uniform_0_1_tensor = uniform_0_1_tensor.reshape(gain_shape)

    #             random_gain_in_db = (
    #                 self.max_gain_db - self.min_gain_db
    #             ) * uniform_0_1_tensor + self.min_gain_db

    #             # Store for logging
    #             if self.mode == "per_example":
    #                 self._last_gain_db = random_gain_in_db.flatten().tolist()
    #             else:
    #                 self._last_gain_db = random_gain_in_db.item()

    #             random_gain_in_a = db_to_amplitude(random_gain_in_db)

    #     return noise_sig * random_gain_in_a

    def forward(self, x):
        noise_sig = torch.empty_like(x).uniform_(generator=self.torch_gen) * 2 - 1

        random_gain_in_a = db_to_amplitude(self._last_gain_db)
        noise_sig = noise_sig * random_gain_in_a

        return x + noise_sig


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
