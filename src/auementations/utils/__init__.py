"""Borrowed from librosa, but converted to torch tensors."""

import warnings
from typing import Callable

import torch
from torch import Tensor


def db_to_power(S_db: Tensor, *, ref: float = 1.0) -> Tensor:
    """Convert dB-scale values to a power values.

    This effectively inverts ``power_to_db``::

        db_to_power(S_db) ~= ref * 10.0**(S_db / 10)

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled values
    ref : number > 0
        Reference power: output will be scaled by this value

    Returns
    -------
    S : np.ndarray
        Power values
    """
    return ref * torch.pow(10.0, S_db * 0.1)


def db_to_amplitude(S_db: Tensor, *, ref: float = 1.0) -> Tensor:
    """Convert a dB-scaled spectrogram to an amplitude spectrogram.

    This effectively inverts `amplitude_to_db`::

        db_to_amplitude(S_db) ~= 10.0**(0.5 * S_db/10 + log10(ref))

    Parameters
    ----------
    S_db : tensor
        dB-scaled values
    ref : number > 0
        Optional reference amplitude.

    Returns
    -------
    S : tensor
        Linear magnitude values
    """
    return db_to_power(S_db, ref=ref**2) ** 0.5


def power_to_db(
    S: Tensor,
    *,
    ref: float | Callable = 1.0,
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> Tensor:
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Parameters
    ----------
    S : np.ndarray
        input power

    ref : scalar or callable
        If scalar, the amplitude ``abs(S)`` is scaled relative to ``ref``::

            10 * log10(S / ref)

        Zeros in the output correspond to positions where ``S == ref``.

        If callable, the reference value is computed as ``ref(S)``.

    amin : float > 0 [scalar]
        minimum threshold for ``abs(S)`` and ``ref``

    top_db : float >= 0 [scalar]
        threshold the output at ``top_db`` below the peak:
        ``max(10 * log10(S/ref)) - top_db``

    Returns
    -------
    S_db : tensor
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``

    Examples
    --------
    Get a power spectrogram from a waveform ``y``

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.power_to_db(S**2)
    array([[-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809],
           ...,
           [-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809]], dtype=float32)

    Compute dB relative to peak power

    >>> librosa.power_to_db(S**2, ref=np.max)
    array([[-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.],
           ...,
           [-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.]], dtype=float32)

    Or compare to median power

    >>> librosa.power_to_db(S**2, ref=np.median)
    array([[16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578],
           ...,
           [16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578]], dtype=float32)

    And plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> imgpow = librosa.display.specshow(S**2, sr=sr, y_axis='log', x_axis='time',
    ...                                   ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> imgdb = librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max),
    ...                                  sr=sr, y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Log-Power spectrogram')
    >>> fig.colorbar(imgpow, ax=ax[0])
    >>> fig.colorbar(imgdb, ax=ax[1], format="%+2.0f dB")
    """
    S = torch.as_tensor(S)

    if amin <= 0:
        raise ValueError("amin must be strictly positive")

    if torch.is_complex(S):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = torch.abs(S)
    else:
        magnitude = S

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = abs(ref)

    log_spec: Tensor = 10.0 * torch.log10(torch.maximum(torch.tensor(amin), magnitude))
    log_spec -= 10.0 * torch.log10(torch.maximum(torch.tensor(amin), torch.tensor(ref_value)))

    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative")
        log_spec = torch.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def amplitude_to_db(
    S: Tensor,
    *,
    ref: float | Callable = 1.0,
    amin: float = 1e-5,
    top_db: float | None = 80.0,
) -> Tensor:
    """Convert an amplitude spectrogram to dB-scaled spectrogram.

    This is equivalent to ``power_to_db(S**2, ref=ref**2, amin=amin**2, top_db=top_db)``,
    but is provided for convenience.

    Parameters
    ----------
    S : np.ndarray
        input amplitude

    ref : scalar or callable
        If scalar, the amplitude ``abs(S)`` is scaled relative to ``ref``:
        ``20 * log10(S / ref)``.
        Zeros in the output correspond to positions where ``S == ref``.

        If callable, the reference value is computed as ``ref(S)``.

    amin : float > 0 [scalar]
        minimum threshold for ``S`` and ``ref``

    top_db : float >= 0 [scalar]
        threshold the output at ``top_db`` below the peak:
        ``max(20 * log10(S/ref)) - top_db``

    Returns
    -------
    S_db : Tensor
        ``S`` measured in dB
    """
    S = torch.as_tensor(S)

    if torch.is_complex(S):
        warnings.warn(
            "amplitude_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call amplitude_to_db(np.abs(S)) instead.",
            stacklevel=2,
        )

    magnitude = torch.abs(S)

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = abs(ref)

    power = torch.square(magnitude)

    db: Tensor = power_to_db(power, ref=ref_value**2, amin=amin**2, top_db=top_db)
    return db
