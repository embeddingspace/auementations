# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AUEM (Audio Unified Extensible Modifications) is a composable audio augmentation library designed for ML training pipelines. It provides a unified interface over multiple audio augmentation backends (torch_audiomentations, pedalboard, etc.) with first-class Hydra/hydra-zen configuration support.

## Development Commands

### Setup and Installation
```bash
# Install dependencies (uses uv)
uv sync

# Install in development mode with all extras
uv pip install -e ".[dev,test,all]"
```

### Testing
```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=auem --cov-report=html

# Run specific test file
uv run pytest tests/test_composition.py

# Run specific test
uv run pytest tests/test_composition.py::test_compose_sequential -v

# Run tests for specific backend
uv run pytest tests/adapters/test_torch_audiomentations.py
```

### Linting and Formatting
```bash
# Format code with ruff
uv run ruff format .

# Lint with ruff
uv run ruff check .

# Fix auto-fixable lint issues
uv run ruff check --fix .

# Type check with mypy
uv run mypy auem
```

## Architecture

### Core Design Principles

1. **Backend Agnostic**: Adapters wrap different augmentation libraries (torch_audiomentations, pedalboard) behind a common interface
2. **Probabilistic by Default**: All parameters support range-based sampling (min/max tuples) for stochastic augmentation
3. **Composition First**: Compose, OneOf, and other combinators are fundamental building blocks
4. **Hydra Native**: All augmentations are instantiable via Hydra configs with structured configs via hydra-zen
5. **Consistent Interface**: Every augmentation receives `sample_rate` explicitly through config

### Directory Structure

- `auem/core/`: Base abstractions and interfaces
  - `base.py`: Abstract base class for all augmentations
  - `composition.py`: Compose, OneOf, SomeOf composition classes
  - `parameters.py`: Parameter sampling utilities (range handling, probability distributions)

- `auem/adapters/`: Backend-specific wrappers
  - `torch_audiomentations.py`: Wraps torch_audiomentations transforms
  - `pedalboard.py`: Wraps Spotify's pedalboard effects
  - Each adapter implements the common `BaseAugmentation` interface

- `auem/config/`: Hydra configuration support
  - `structured.py`: Hydra-zen structured configs for all augmentations
  - `schemas.py`: Dataclasses and validation schemas

- `auem/utils/`: Shared utilities
  - `audio.py`: Audio I/O and manipulation helpers
  - `validation.py`: Input validation and type checking

- `examples/`: Usage demonstrations
  - `basic_usage.py`: Simple augmentation examples
  - `hydra_config/`: Example Hydra config YAML files
  - `training_pipeline.py`: Integration with ML training loop

- `tests/`: Test suite organized to mirror source structure

### Key Abstractions

**BaseAugmentation**: All augmentations inherit from this. Key methods:
- `__call__(audio, sample_rate)`: Apply augmentation to audio tensor/array
- `randomize_parameters()`: Sample parameters from configured ranges
- `to_config()`: Export to Hydra-compatible config dict

**Parameter Sampling**: Parameters can be specified as:
- Single values: `gain=0.5`
- Ranges (uniform): `gain=(0.0, 1.0)`
- Distributions: `gain={"dist": "log_uniform", "min": 0.01, "max": 1.0}`

**Composition**:
- `Compose`: Sequential application (all augmentations applied in order)
- `OneOf`: Choose one augmentation randomly
- `SomeOf`: Apply k random augmentations from n options

### Backend Adapter Pattern

Each backend adapter:
1. Wraps the underlying library's augmentation class
2. Translates AUEM's parameter format to the backend's expected format
3. Handles tensor/array conversions between backends
4. Ensures `sample_rate` is properly passed through

Example adapter structure:
```python
class TorchAudiomentationsAdapter(BaseAugmentation):
    def __init__(self, augmentation_cls, sample_rate, **params):
        self.sample_rate = sample_rate
        self._aug = augmentation_cls(**self._resolve_params(params))

    def __call__(self, audio):
        return self._aug(audio, sample_rate=self.sample_rate)
```

### Hydra Integration

All augmentations expose structured configs:
- Use `hydra_zen.builds()` to create structured configs
- Support nested composition through config groups
- Config instantiation via `hydra.utils.instantiate()`

Config organization:
- `conf/augmentation/`: Individual augmentation configs
- `conf/augmentation/compose/`: Composition pipeline configs
- `conf/augmentation/adapter/`: Backend-specific settings

### Testing Strategy

**TDD Workflow (RED, GREEN, REFACTOR)**:
- Always write tests BEFORE implementation code
- RED: Write a failing test that defines the desired behavior
- GREEN: Write minimal code to make the test pass
- REFACTOR: Clean up and optimize while keeping tests green
- Never write production code without a failing test first

**BDD Test Structure (GIVEN, WHEN, THEN)**:
Tests should follow the Behavior-Driven Development pattern where practical:
```python
def test_gain_augmentation_applies_specified_gain():
    # GIVEN: An audio sample and a gain augmentation with fixed gain
    audio = torch.randn(1, 16000)
    sample_rate = 16000
    gain_db = 6.0
    augmentation = Gain(gain_db=gain_db, sample_rate=sample_rate)

    # WHEN: The augmentation is applied
    result = augmentation(audio)

    # THEN: The output amplitude is increased by the specified gain
    expected_gain_linear = 10 ** (gain_db / 20)
    assert torch.allclose(result, audio * expected_gain_linear, atol=1e-5)
```

**Test Types**:
- Unit tests for each augmentation adapter (mock backend calls)
- Integration tests with actual audio (fixtures in `tests/fixtures/audio/`)
- Composition tests ensure proper chaining and probability handling
- Hydra config tests validate instantiation from YAML
- Parametric tests across backends to ensure consistent behavior

### Important Implementation Notes

- **Sample Rate Consistency**: Always pass `sample_rate` explicitly; never infer or assume
- **Tensor Type Handling**: Adapters handle torch.Tensor ↔ numpy.ndarray conversions transparently
- **Probability Semantics**: `p=0.5` means 50% chance to apply, consistent with torch_audiomentations
- **Reproducibility**: All random sampling should respect `torch.manual_seed()` / `numpy.random.seed()`
- **Lazy Loading**: Backend libraries imported only when their adapter is instantiated (avoid requiring all deps)
