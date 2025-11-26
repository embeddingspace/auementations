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
import numpy as np

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

# Apply to audio (numpy array or torch tensor)
audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
augmented = augment(audio)
```

### Probabilistic Parameters

Parameters can be specified as ranges for random sampling:

```python
from auementations.adapters.pedalboard import LowPassFilter

# Range-based sampling (uniform distribution)
lpf = LowPassFilter(
    sample_rate=44100,
    min_cutoff_freq=500.0,     # Randomly sampled each call
    max_cutoff_freq=5000.0,    # Randomly sampled each call
    p=0.7                      # 70% chance to apply
)

# Or use fixed parameter
lpf_fixed = LowPassFilter(
    sample_rate=44100,
    cutoff_freq=1000.0,  # Always 1000 Hz
    p=1.0
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
        AddColoredNoise(sample_rate=16000, min_snr_db=3, max_snr_db=30)
    ]
)

# Sequential composition
pipeline = Compose([one_of, some_of])
```

### Available Backends

**torch_audiomentations**: GPU-accelerated augmentations
- `Gain`: Adjust volume
- `PitchShift`: Shift pitch up/down
- `TimeStretch`: Change speed without affecting pitch
- `AddColoredNoise`: Add white/pink/brown noise
- `HighPassFilter`: Remove low frequencies
- `LowPassFilter`: Remove high frequencies

**pedalboard**: High-quality audio effects from Spotify
- `LowPassFilter`: Remove high frequencies
- `HighPassFilter`: Remove low frequencies
- More effects coming soon!

### Hydra Configuration

Auementations provides first-class Hydra integration via hydra-zen:

```python
from hydra_zen import builds, instantiate
from auementations.config.config_store import auementations_store
from auementations.core.composition import OneOf

# Get configs from the centralized store
lpf_config = auementations_store.get_entry(
    group="augmentation/pedalboard", name="lpf"
)["node"]
hpf_config = auementations_store.get_entry(
    group="augmentation/pedalboard", name="hpf"
)["node"]
one_of_config = auementations_store.get_entry(
    group="augmentation/composition", name="one_of"
)["node"]

# Build a structured config
composition_config = builds(
    one_of_config,
    augmentations=[
        builds(
            lpf_config,
            sample_rate=16000,
            min_cutoff_freq=500.0,
            max_cutoff_freq=2000.0,
        ),
        builds(
            hpf_config,
            sample_rate=16000,
            min_cutoff_freq=100.0,
            max_cutoff_freq=500.0,
        ),
    ],
    sample_rate=16000,
    p=1.0,
)

# Instantiate and use
augmentation = instantiate(composition_config)
augmented = augmentation(audio)
```

Or use dictionary-based configs (like from YAML):

```python
from hydra_zen import instantiate

config_dict = {
    "_target_": "auementations.core.composition.OneOf",
    "augmentations": [
        {
            "_target_": "auementations.adapters.pedalboard.LowPassFilter",
            "sample_rate": 16000,
            "min_cutoff_freq": 500.0,
            "max_cutoff_freq": 2000.0,
        },
        {
            "_target_": "auementations.adapters.torch_audiomentations.Gain",
            "sample_rate": 16000,
            "min_gain_db": -12.0,
            "max_gain_db": 12.0,
        },
    ],
    "sample_rate": 16000,
}

augmentation = instantiate(config_dict)
```

### Multi-Backend Usage

Mix and match augmentations from different backends:

```python
from auementations.adapters.torch_audiomentations import Gain, AddColoredNoise
from auementations.adapters.pedalboard import LowPassFilter, HighPassFilter
from auementations.core.composition import Compose, OneOf

# Combine torch_audiomentations and pedalboard
pipeline = Compose([
    Gain(sample_rate=44100, min_gain_db=-6, max_gain_db=6, p=0.5),
    OneOf([
        LowPassFilter(sample_rate=44100, min_cutoff_freq=500, max_cutoff_freq=5000),
        HighPassFilter(sample_rate=44100, min_cutoff_freq=100, max_cutoff_freq=1000),
    ], p=0.7),
    AddColoredNoise(sample_rate=44100, min_snr_db=10, max_snr_db=30, p=0.4),
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

1. **Write tests first** (TDD): Create `tests/adapters/test_your_backend.py` following BDD Given-When-Then pattern
2. **Create adapter**: Implement in `auementations/adapters/your_backend.py`, inheriting from `BaseAugmentation`
3. **Register configs**: Use `@auementations_store` decorator to register with the centralized store
4. **Export classes**: Add to `auementations/adapters/__init__.py`
5. **Update config store**: Import in `auementations/config/config_store.py`

See `auementations/adapters/pedalboard.py` and `tests/adapters/test_pedalboard.py` for reference implementation.

## License

MIT License - see LICENSE file for details
