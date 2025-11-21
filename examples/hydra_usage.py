"""Example of using Auementations with Hydra for configuration management."""

import numpy as np
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def run_with_hydra_config():
    """Load and use augmentation pipeline from Hydra config."""
    print("=" * 60)
    print("Hydra Configuration Example")
    print("=" * 60)

    # Initialize Hydra with the config directory
    with initialize(version_base=None, config_path="hydra_config"):
        # Compose configuration
        cfg = compose(config_name="config")

        # Print the loaded configuration
        print("\nLoaded configuration:")
        print(OmegaConf.to_yaml(cfg.augmentation))

        # Instantiate the augmentation pipeline
        augmentation = instantiate(cfg.augmentation)

        print(f"\nInstantiated pipeline: {type(augmentation).__name__}")
        print(f"Sample rate: {augmentation.sample_rate}")
        print(f"Number of augmentations: {len(augmentation.augmentations)}")

        # Create sample audio
        sample_rate = cfg.data.sample_rate
        duration = cfg.data.duration
        num_samples = int(sample_rate * duration)

        audio = np.random.randn(num_samples).astype(np.float32)

        # Apply augmentation pipeline
        print(f"\nApplying augmentation to audio of shape {audio.shape}")

        n_trials = 5
        print(f"\nRunning {n_trials} trials:")
        for i in range(n_trials):
            augmented = augmentation(audio.copy())
            gain = np.abs(augmented[100] / audio[100])
            print(f"  Trial {i+1}: Gain = {gain:.2f}x")


def programmatic_config_with_hydra_zen():
    """Create augmentation config programmatically using hydra-zen."""
    print("\n" + "=" * 60)
    print("Programmatic Config with hydra-zen")
    print("=" * 60)

    from hydra_zen import builds, instantiate
    from auementations.core.composition import Compose, OneOf
    from tests.conftest import MockAugmentation

    # Build configs using hydra-zen
    Aug1Config = builds(
        MockAugmentation,
        sample_rate=16000,
        gain=1.5,
        p=1.0,
        populate_full_signature=True
    )

    Aug2Config = builds(
        MockAugmentation,
        sample_rate=16000,
        gain=2.0,
        p=1.0,
        populate_full_signature=True
    )

    OneOfConfig = builds(
        OneOf,
        augmentations=[Aug1Config, Aug2Config],
        sample_rate=16000,
        weights=[0.7, 0.3],
        p=1.0,
        populate_full_signature=True
    )

    # Instantiate
    one_of = instantiate(OneOfConfig)

    # Create audio and apply
    audio = np.random.randn(16000).astype(np.float32)

    print(f"\nCreated OneOf with 2 augmentations")
    print(f"Weights: [0.7, 0.3]")
    print("\nApplying 10 times:")

    gain_counts = {1.5: 0, 2.0: 0}
    for i in range(10):
        augmented = one_of(audio.copy())
        gain = augmented[100] / audio[100]
        gain_rounded = round(gain, 1)
        gain_counts[gain_rounded] = gain_counts.get(gain_rounded, 0) + 1
        print(f"  Trial {i+1}: Gain = {gain:.1f}x")

    print(f"\nGain distribution:")
    for gain, count in gain_counts.items():
        print(f"  {gain}x: {count} times ({count/10*100:.0f}%)")


def override_config_from_command_line():
    """Example showing how to override config values."""
    print("\n" + "=" * 60)
    print("Config Override Example")
    print("=" * 60)

    # Initialize with overrides
    with initialize(version_base=None, config_path="hydra_config"):
        # Override sample_rate and probability
        cfg = compose(
            config_name="config",
            overrides=[
                "data.sample_rate=44100",
                "augmentation.p=0.5",
            ]
        )

        print("\nOverridden configuration:")
        print(f"  data.sample_rate: {cfg.data.sample_rate}")
        print(f"  augmentation.p: {cfg.augmentation.p}")

        # Instantiate with overridden values
        augmentation = instantiate(cfg.augmentation)
        print(f"\nPipeline sample_rate: {augmentation.sample_rate}")
        print(f"Pipeline probability: {augmentation.p}")


if __name__ == "__main__":
    # Run examples
    run_with_hydra_config()
    programmatic_config_with_hydra_zen()
    override_config_from_command_line()

    print("\n" + "=" * 60)
    print("All Hydra examples completed!")
    print("=" * 60)
