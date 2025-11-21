# Auementations

A composable audio augmentation library designed for ML training pipelines with first-class Hydra configuration support.

## Features

- **Unified Interface**: Common API across multiple audio augmentation backends (torch_audiomentations, pedalboard, etc.)
- **Probabilistic Augmentations**: Built-in support for parameter ranges and probability distributions
- **Composition First**: Compose, OneOf, SomeOf combinators for flexible augmentation pipelines
- **Hydra Native**: Full integration with Hydra/hydra-zen for configuration management
- **Type Safe**: Comprehensive type hints and runtime validation
- **Extensible**: Easy to add new backends and custom augmentations

## Installation

```bash
# Basic installation
pip install auementations

# With specific backends
pip install auementations[torch]
pip install auementations[pedalboard]

# All backends
pip install auumentations[all]

# Development
uv sync
```

## Quick Start

### Basic Usage

```python
from auementations.adapters.torch_audiomentations import Gain, PitchShift
from auementations.core.composition import Compose

# Create augmentation pipeline
augment = Compose([
    Gain(
        sample_rate=16000,
        min_gain_db=-12.0,
        max_gain_db=12.0,
        p=0.5
    ),
    PitchShift(
        sample_rate=16000,
        min_semitones=-4,
        max_semitones=4,
        p=0.3
    )
])

# Apply to audio
augmented = augment(audio_tensor)
```

### Probabilistic Parameters

Parameters can be specified as ranges for random sampling:

```python
from auementations.adapters.pedalboard import Reverb

# Range-based sampling (uniform distribution)
reverb = Reverb(
    sample_rate=44100,
    room_size=(0.1, 0.9),      # Randomly sampled each call
    wet_level=(0.1, 0.5),       # Randomly sampled each call
    p=0.7                       # 70% chance to apply
)
```

### Composition Combinators

```python
from auementations.core.composition import OneOf, SomeOf, Compose
from auementations.adapters.torch_audiomentations import *

# Apply exactly one augmentation randomly
one_of = OneOf([
    TimeStretch(sample_rate=16000, min_rate=0.8, max_rate=1.2),
    PitchShift(sample_rate=16000, min_semitones=-2, max_semitones=2),
    AddColoredNoise(sample_rate=16000, min_snr_db=3, max_snr_db=30)
])

# Apply 2 random augmentations from the list
some_of = SomeOf(
    k=2,
    augmentations=[
        Gain(sample_rate=16000, min_gain_db=-6, max_gain_db=6),
        HighPassFilter(sample_rate=16000, min_cutoff_freq=20, max_cutoff_freq=400),
        LowPassFilter(sample_rate=16000, min_cutoff_freq=4000, max_cutoff_freq=8000),
        AddBackgroundNoise(sample_rate=16000, min_snr_db=3, max_snr_db=30)
    ]
)

# Sequential composition
pipeline = Compose([one_of, some_of])
```

### Hydra Configuration

```yaml
# config/augmentation/train_pipeline.yaml
_target_: auementations.core.composition.Compose
augmentations:
  - _target_: auementations.adapters.torch_audiomentations.Gain
    sample_rate: ${data.sample_rate}
    min_gain_db: -12.0
    max_gain_db: 12.0
    p: 0.5

  - _target_: auementations.core.composition.OneOf
    augmentations:
      - _target_: auementations.adapters.torch_audiomentations.PitchShift
        sample_rate: ${data.sample_rate}
        min_semitones: -4
        max_semitones: 4

      - _target_: auementations.adapters.pedalboard.Reverb
        sample_rate: ${data.sample_rate}
        room_size: [0.1, 0.9]
        wet_level: [0.1, 0.5]
```

```python
# In your training code
from hydra import compose, initialize
from hydra.utils import instantiate

with initialize(config_path="config"):
    cfg = compose(config_name="train")
    augmentation = instantiate(cfg.augmentation)

    # Use in training loop
    augmented_audio = augmentation(audio, sample_rate=cfg.data.sample_rate)
```

### Multi-Backend Usage

Mix and match augmentations from different backends:

```python
from auementations.adapters.torch_audiomentations import Gain, AddColoredNoise
from auementations.adapters.pedalboard import Reverb, Chorus
from auementations.core.composition import Compose

# Combine torch_audiomentations and pedalboard
pipeline = Compose([
    Gain(sample_rate=44100, min_gain_db=-6, max_gain_db=6, p=0.5),
    Reverb(sample_rate=44100, room_size=(0.2, 0.8), p=0.3),
    AddColoredNoise(sample_rate=44100, min_snr_db=10, max_snr_db=30, p=0.4),
    Chorus(sample_rate=44100, rate_hz=(0.5, 2.0), p=0.2)
])
```

## Architecture

Auementations provides:

1. **Base Abstraction** (`auumentations.core.base.BaseAugmentation`): Common interface for all augmentations
2. **Composition Tools** (`auumentations.core.composition`): Compose, OneOf, SomeOf for building pipelines
3. **Parameter Sampling** (`auumentations.core.parameters`): Flexible parameter randomization
4. **Backend Adapters** (`auumentations.adapters.*`): Wrappers for torch_audiomentations, pedalboard, etc.
5. **Hydra Integration** (`auumentations.config`): Structured configs for all components

## Design Philosophy

- **Explicit sample_rate**: Always specify sample rate in configuration, never infer
- **Lazy backend loading**: Only import backend libraries when their adapters are used
- **Reproducible randomness**: Respect global random seeds for reproducibility
- **Type safety**: Comprehensive type hints and validation
- **Configuration over code**: Prefer Hydra configs for pipeline definition

## Development

```bash
# Setup development environment
uv sync

# Run tests
uv run pytest

# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy auementations
```

## Contributing

Contributions welcome! To add a new backend:

1. Create adapter in `auem/adapters/your_backend.py`
2. Inherit from `BaseAugmentation`
3. Implement required methods
4. Add tests in `tests/adapters/test_your_backend.py`
5. Add structured configs in `auem/config/structured.py`

## License

MIT License - see LICENSE file for details
