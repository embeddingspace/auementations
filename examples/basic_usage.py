"""Basic usage examples for Auementations."""

import numpy as np
from auementations.core.composition import Compose, OneOf, SomeOf
from tests.conftest import MockAugmentation


def example_simple_augmentation():
    """Example: Apply a single augmentation."""
    print("=" * 60)
    print("Example 1: Simple Augmentation")
    print("=" * 60)

    # Create audio signal
    audio = np.random.randn(16000).astype(np.float32)

    # Create augmentation
    aug = MockAugmentation(sample_rate=16000, gain=2.0, p=1.0)

    # Apply
    augmented = aug(audio)

    print(f"Original audio shape: {audio.shape}")
    print(f"Augmented audio shape: {augmented.shape}")
    print(f"Gain applied: {augmented[0] / audio[0]:.1f}x")
    print()


def example_sequential_composition():
    """Example: Compose multiple augmentations sequentially."""
    print("=" * 60)
    print("Example 2: Sequential Composition (Compose)")
    print("=" * 60)

    # Create audio
    audio = np.random.randn(16000).astype(np.float32)

    # Create pipeline: apply gain, then another gain
    pipeline = Compose(
        [
            MockAugmentation(sample_rate=16000, gain=2.0, p=1.0),
            MockAugmentation(sample_rate=16000, gain=1.5, p=1.0),
        ]
    )

    # Apply
    augmented = pipeline(audio)

    print("Pipeline: Gain(2.0) -> Gain(1.5)")
    print("Expected total gain: 2.0 * 1.5 = 3.0x")
    print(f"Actual gain: {augmented[100] / audio[100]:.1f}x")
    print()


def example_one_of_selection():
    """Example: Randomly select one augmentation from a list."""
    print("=" * 60)
    print("Example 3: Random Selection (OneOf)")
    print("=" * 60)

    # Create audio
    audio = np.random.randn(16000).astype(np.float32)

    # Create OneOf: choose one augmentation randomly
    one_of = OneOf(
        [
            MockAugmentation(sample_rate=16000, gain=2.0, p=1.0),
            MockAugmentation(sample_rate=16000, gain=3.0, p=1.0),
            MockAugmentation(sample_rate=16000, gain=4.0, p=1.0),
        ]
    )

    # Apply multiple times to see different selections
    print("Applying OneOf 5 times:")
    for i in range(5):
        augmented = one_of(audio.copy())
        gain = augmented[100] / audio[100]
        print(f"  Trial {i + 1}: Gain = {gain:.1f}x")
    print()


def example_weighted_selection():
    """Example: OneOf with weights."""
    print("=" * 60)
    print("Example 4: Weighted Selection")
    print("=" * 60)

    # Create audio
    audio = np.random.randn(16000).astype(np.float32)

    # Create OneOf with weights: 70% gain=2.0, 30% gain=3.0
    one_of = OneOf(
        augmentations=[
            MockAugmentation(sample_rate=16000, gain=2.0, p=1.0),
            MockAugmentation(sample_rate=16000, gain=3.0, p=1.0),
        ],
        weights=[0.7, 0.3],
    )

    # Apply many times and count
    n_trials = 1000
    gain_2_count = 0
    gain_3_count = 0

    for _ in range(n_trials):
        augmented = one_of(audio.copy())
        gain = augmented[100] / audio[100]
        if abs(gain - 2.0) < 0.1:
            gain_2_count += 1
        else:
            gain_3_count += 1

    print(f"Applied OneOf {n_trials} times with weights [0.7, 0.3]:")
    print(f"  Gain 2.0x: {gain_2_count} times ({gain_2_count / n_trials * 100:.1f}%)")
    print(f"  Gain 3.0x: {gain_3_count} times ({gain_3_count / n_trials * 100:.1f}%)")
    print()


def example_some_of():
    """Example: Apply k random augmentations."""
    print("=" * 60)
    print("Example 5: Apply K Random Augmentations (SomeOf)")
    print("=" * 60)

    # Create audio
    audio = np.random.randn(16000).astype(np.float32)

    # Create SomeOf: apply exactly 2 augmentations
    some_of = SomeOf(
        k=2,
        augmentations=[
            MockAugmentation(sample_rate=16000, gain=1.5, p=1.0),
            MockAugmentation(sample_rate=16000, gain=1.3, p=1.0),
            MockAugmentation(sample_rate=16000, gain=1.2, p=1.0),
            MockAugmentation(sample_rate=16000, gain=1.1, p=1.0),
        ],
        replace=False,
    )

    print("Applying SomeOf(k=2) from 4 augmentations, 3 times:")
    for i in range(3):
        augmented = some_of(audio.copy())
        gain = augmented[100] / audio[100]
        print(f"  Trial {i + 1}: Total gain = {gain:.2f}x")
    print()


def example_nested_composition():
    """Example: Nested composition (Compose of OneOf)."""
    print("=" * 60)
    print("Example 6: Nested Composition")
    print("=" * 60)

    # Create audio
    audio = np.random.randn(16000).astype(np.float32)

    # Create complex pipeline:
    # 1. Always apply gain of 2.0
    # 2. Then randomly choose between two augmentations
    pipeline = Compose(
        [
            MockAugmentation(sample_rate=16000, gain=2.0, p=1.0),
            OneOf(
                [
                    MockAugmentation(sample_rate=16000, gain=1.5, p=1.0),
                    MockAugmentation(sample_rate=16000, gain=2.0, p=1.0),
                ]
            ),
        ]
    )

    print("Pipeline: Always(Gain 2.0) -> OneOf(Gain 1.5 or 2.0)")
    print("Applying 5 times:")
    for i in range(5):
        augmented = pipeline(audio.copy())
        gain = augmented[100] / audio[100]
        print(f"  Trial {i + 1}: Total gain = {gain:.1f}x")
    print()


def example_probabilistic_application():
    """Example: Probabilistic application with p parameter."""
    print("=" * 60)
    print("Example 7: Probabilistic Application")
    print("=" * 60)

    # Create audio
    audio = np.random.randn(16000).astype(np.float32)

    # Create augmentation with 50% probability
    aug = MockAugmentation(sample_rate=16000, gain=2.0, p=0.5)

    # Apply many times
    n_trials = 1000
    applied_count = 0

    for _ in range(n_trials):
        augmented = aug(audio.copy())
        if not np.array_equal(augmented, audio):
            applied_count += 1

    print(f"Augmentation with p=0.5 applied {n_trials} times:")
    print(
        f"  Actually applied: {applied_count} times ({applied_count / n_trials * 100:.1f}%)"
    )
    print("  Expected: ~500 times (50%)")
    print()


def example_config_export():
    """Example: Export augmentation to config dict."""
    print("=" * 60)
    print("Example 8: Export to Config")
    print("=" * 60)

    # Create complex pipeline
    pipeline = Compose(
        [
            MockAugmentation(sample_rate=16000, gain=2.0, p=0.8),
            OneOf(
                [
                    MockAugmentation(sample_rate=16000, gain=1.5, p=1.0),
                    MockAugmentation(sample_rate=16000, gain=2.0, p=1.0),
                ],
                weights=[0.6, 0.4],
            ),
        ]
    )

    # Export to config
    config = pipeline.to_config()

    print("Exported pipeline config:")
    print(f"  _target_: {config['_target_']}")
    print(f"  sample_rate: {config['sample_rate']}")
    print(f"  p: {config['p']}")
    print(f"  Number of augmentations: {len(config['augmentations'])}")
    print()


if __name__ == "__main__":
    # Run all examples
    example_simple_augmentation()
    example_sequential_composition()
    example_one_of_selection()
    example_weighted_selection()
    example_some_of()
    example_nested_composition()
    example_probabilistic_application()
    example_config_export()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
