## 0.3.0 (2025-12-22)

### Feat

- biquad/peaking filter augmentations
- NoiseAugmentation with per_example, per_batch, and fixed or range modes
- auementations gain
- auementations gain
- gain augmentation with settings per batch/example/source/channel
- handle additional dimensions
- allow pedalboard to handle dims of 3,4

### Fix

- make sure pedalboard can't infinitify things
- fix alls, imports, and nyquist
- adds ruff, fixes lints, fixes bugs from lints
- lpf and hpf augmentations were being applied to the whole batch! now per example
- adds auementation store for gain
- remove unnecessary code
- fix groups in tests
- update auementations store group for consistency

## 0.2.3 (2025-11-26)

### Fix

- remove composition via lists; it's messing with hydra

## 0.2.2 (2025-11-26)

### Fix

- didn't handle sampleing rate quite far enough down in the chain

## 0.2.1 (2025-11-26)

### Fix

- allow samplerate to be float

## 0.2.0 (2025-11-26)

### Feat

- compose with dicts
- adds pedalboard adapters for lpf/hpf
- adds working config store
- initial repo commit/setup

### Refactor

- remove multi-level group
- move code into src
- config store to use decorator for registration
