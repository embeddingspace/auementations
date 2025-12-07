import torch
import torch.nn as nn

from auementations.utils import amplitude_to_db, db_to_amplitude


class GainAugmentation(nn.Module):
    VALID_MODES = ["per_batch", "per_example", "per_source", "per_channel"]

    def __init__(
        self,
        sample_rate: int | float | None = None,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        p: float = 1.0,
        mode: str = "per_example",
        seed: int | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.p = p
        self.seed = seed

        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.mode = mode

        self.generator = torch.Generator()
        if self.seed is not None:
            self.generator.manual_seed(self.seed)

        if sample_rate is not None and sample_rate <= 0:
            raise ValueError(f"sample_rate must be an int or float > 0.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_apply = torch.rand((), generator=self.generator)

        if p_apply <= self.p:
            # Determine the reduction dimensions and gain shape based on mode
            match self.mode:
                case "per_example":
                    # One gain per example (batch dimension)
                    num_gains = x.shape[0]
                    gain_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                    # Reduce over all dimensions except batch
                    dims_to_reduce = tuple(range(1, x.ndim))
                    signal_max = x.abs().amax(dim=dims_to_reduce, keepdim=False)
                case "per_source":
                    # One gain per source (dimension 1)
                    if x.ndim < 2:
                        raise ValueError(
                            f"per_source mode requires at least 2 dimensions, got {x.ndim}"
                        )
                    num_gains = x.shape[0] * x.shape[1]
                    gain_shape = (x.shape[0], x.shape[1]) + (1,) * (x.ndim - 2)
                    # Reduce over all dimensions except batch and source
                    dims_to_reduce = tuple(range(2, x.ndim))
                    signal_max = (
                        x.abs().amax(dim=dims_to_reduce, keepdim=False).flatten()
                    )
                case "per_channel":
                    # One gain per channel (dimension 2)
                    if x.ndim < 3:
                        raise ValueError(
                            f"per_channel mode requires at least 3 dimensions, got {x.ndim}"
                        )
                    num_gains = x.shape[0] * x.shape[1] * x.shape[2]
                    gain_shape = (x.shape[0], x.shape[1], x.shape[2]) + (1,) * (
                        x.ndim - 3
                    )
                    # Reduce over all dimensions except batch, source, and channel
                    dims_to_reduce = tuple(range(3, x.ndim))
                    signal_max = (
                        x.abs().amax(dim=dims_to_reduce, keepdim=False).flatten()
                    )
                # case "per_batch":
                case _:
                    # Single gain for entire batch
                    num_gains = 1
                    gain_shape = (1,) * x.ndim  # Will broadcast to all dimensions
                    # Reduce over all dimensions
                    signal_max = x.abs().max()

            # Sample random gain values
            uniform_0_1_tensor = torch.rand((num_gains,), generator=self.generator)

            # Prevent gain from going > 1.0 by clipping the gain max
            # Convert signal max to dB and compute max allowable gain
            signal_max_db = amplitude_to_db(signal_max)

            if self.mode == "per_batch":
                # Scalar case - keep as tensor
                max_gain_db_scalar = min(self.max_gain_db, signal_max_db.item() * -1)
                max_gain_db = torch.tensor(
                    max_gain_db_scalar,
                    dtype=signal_max_db.dtype,
                    device=signal_max_db.device,
                )
            else:
                # Vectorized version for multiple gains
                max_gain_db = torch.minimum(
                    torch.tensor(
                        self.max_gain_db,
                        dtype=signal_max_db.dtype,
                        device=signal_max_db.device,
                    ),
                    signal_max_db * -1,
                )

            # Calculate random gain in dB (works for both scalar and vector cases)
            random_gain_in_db = (
                max_gain_db - self.min_gain_db
            ) * uniform_0_1_tensor + self.min_gain_db

            # Convert to amplitude and reshape
            random_gain_in_a = db_to_amplitude(random_gain_in_db)
            random_gain_in_a = random_gain_in_a.reshape(gain_shape)

            # Apply gain
            x = x * random_gain_in_a

        return x
