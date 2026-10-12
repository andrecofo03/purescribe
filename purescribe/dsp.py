"""Frame-level signal processing: pre-emphasis, framing, windowing, spectra.

Speech is only wide-sense stationary over roughly 20-30 ms, so every analysis
here begins by cutting the signal into short overlapping frames and treating
each one as if it came from a fixed source-filter configuration.
"""

from __future__ import annotations

import numpy as np

PREEMPHASIS = 0.97


def preemphasis(x: np.ndarray, coeff: float = PREEMPHASIS) -> np.ndarray:
    """First-order high-pass: y[n] = x[n] - a*x[n-1].

    The glottal source falls off at about -12 dB/octave and lip radiation adds
    back +6, leaving roughly -6 dB/octave across the spectrum. One zero near
    z = 0.97 flattens that tilt, so the high formants are not buried under F1
    when the filterbank integrates energy.
    """
    if coeff == 0.0 or x.size == 0:
        return np.asarray(x, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    out[0] = x[0]
    out[1:] = x[1:] - coeff * x[:-1]
    return out


def frame_signal(
    x: np.ndarray,
    frame_length: int,
    hop_length: int,
    pad: bool = True,
) -> np.ndarray:
    """Cut `x` into overlapping frames -> shape (n_frames, frame_length).

    Uses a strided view, so no sample is copied until the window is applied.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < frame_length:
        if not pad:
            return np.zeros((0, frame_length))
        x = np.pad(x, (0, frame_length - x.size))

    if pad:
        # Pad the tail so the final partial frame is kept rather than dropped.
        remainder = (x.size - frame_length) % hop_length
        if remainder:
            x = np.pad(x, (0, hop_length - remainder))

    n_frames = 1 + (x.size - frame_length) // hop_length
    stride = x.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_length),
        strides=(stride * hop_length, stride),
        writeable=False,
    )
    return frames


def window(frames: np.ndarray, kind: str = "hamming") -> np.ndarray:
    """Apply an analysis window to each frame.

    Rectangular framing convolves the spectrum with a sinc whose sidelobes fall
    off at only -13 dB, smearing formants. Hamming drops the first sidelobe to
    about -43 dB at the cost of a wider main lobe - the right trade here,
    because we integrate into broad mel bands anyway.
    """
    n = frames.shape[-1]
    if kind == "hamming":
        w = np.hamming(n)
    elif kind == "hann":
        w = np.hanning(n)
    elif kind == "rect":
        w = np.ones(n)
    else:
        raise ValueError(f"unknown window: {kind}")
    return frames * w


def power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """|FFT|^2 / N for each frame -> shape (n_frames, n_fft//2 + 1).

    Phase is discarded. The ear is largely insensitive to it within a frame,
    and dropping it makes the representation shift-invariant inside the window.
    """
    spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    return (np.abs(spec) ** 2) / n_fft


def frame_energy(frames: np.ndarray) -> np.ndarray:
    """Short-time energy, one value per frame."""
    return np.sum(frames.astype(np.float64) ** 2, axis=-1)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Fraction of adjacent sample pairs that change sign.

    A cheap proxy for spectral centre of mass: fricatives cross zero often,
    voiced vowels rarely. Pairs with energy to separate silence from speech.
    """
    signs = np.signbit(frames)
    return np.mean(signs[:, 1:] != signs[:, :-1], axis=-1)


def stft_db(
    x: np.ndarray,
    frame_length: int = 400,
    hop_length: int = 160,
    n_fft: int = 512,
    floor_db: float = -100.0,
) -> np.ndarray:
    """Log-power spectrogram in dB. For inspection and plotting."""
    frames = window(frame_signal(preemphasis(x), frame_length, hop_length))
    p = power_spectrum(frames, n_fft)
    db = 10.0 * np.log10(np.maximum(p, 1e-30))
    return np.maximum(db, db.max() + floor_db if db.size else floor_db)
