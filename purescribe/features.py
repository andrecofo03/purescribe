"""MFCC front end: the representation the acoustic model actually sees.

The chain is FFT power -> mel filterbank -> log -> DCT -> liftering, followed
by delta regression and cepstral mean/variance normalisation. Each step has a
specific job, documented at the function that performs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fftpack import dct

from .dsp import (
    frame_energy,
    frame_signal,
    power_spectrum,
    preemphasis,
    window,
)


def hz_to_mel(f):
    """O'Shaughnessy's mel scale: linear below ~1 kHz, logarithmic above."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    n_filters: int,
    n_fft: int,
    rate: int,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Triangular filters equally spaced on the mel scale -> (n_filters, n_fft//2+1).

    Critical-band spacing mirrors cochlear resolution: fine at low frequency
    where formants crowd together, coarse up high where only gross spectral
    shape matters. Integrating over a band also smooths away pitch harmonics,
    which is exactly what we want - the phoneme is in the envelope, not F0.
    """
    if fmax is None:
        fmax = rate / 2.0
    fmax = min(fmax, rate / 2.0)

    # n_filters triangles need n_filters+2 edge points: each filter spans
    # [edge[i], edge[i+2]] and peaks at edge[i+1].
    mel_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
    hz_edges = mel_to_hz(mel_edges)

    freqs = np.linspace(0, rate / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_filters, freqs.size))
    for i in range(n_filters):
        left, centre, right = hz_edges[i], hz_edges[i + 1], hz_edges[i + 2]
        if right <= left:
            continue
        rising = (freqs - left) / max(centre - left, 1e-10)
        falling = (right - freqs) / max(right - centre, 1e-10)
        fb[i] = np.maximum(0.0, np.minimum(rising, falling))
    return fb


def lifter(cepstra: np.ndarray, L: int = 22) -> np.ndarray:
    """Sinusoidal liftering: rescale cepstral coefficients by index.

    Higher-order cepstra have small dynamic range; boosting them stops the
    first few coefficients from dominating the Euclidean-style distances that
    a diagonal Gaussian implicitly computes.
    """
    if L <= 0:
        return cepstra
    n = np.arange(cepstra.shape[-1])
    w = 1.0 + (L / 2.0) * np.sin(np.pi * n / L)
    return cepstra * w


def delta(feat: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression-based time derivative over +/- `width` frames.

    An HMM assumes each frame is conditionally independent given its state,
    which is plainly false for speech. Stacking derivatives smuggles the local
    trajectory into the observation vector and repairs much of that damage.
    """
    if feat.shape[0] == 0:
        return feat.copy()
    # Edge frames are replicated so the regression window is always full.
    padded = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    denom = 2.0 * sum(n * n for n in range(1, width + 1))
    out = np.zeros_like(feat)
    for n in range(1, width + 1):
        out += n * (padded[width + n : width + n + feat.shape[0]]
                    - padded[width - n : width - n + feat.shape[0]])
    return out / denom


def cmvn(feat: np.ndarray, variance: bool = True) -> np.ndarray:
    """Cepstral mean (and variance) normalisation over the utterance.

    A fixed linear channel - microphone, room, codec - is a convolution in time,
    so it becomes an additive constant in the log-spectral and hence cepstral
    domain. Subtracting the utterance mean removes it outright. This is the
    single most valuable robustness step in the whole front end.
    """
    if feat.shape[0] == 0:
        return feat.copy()
    out = feat - feat.mean(axis=0, keepdims=True)
    if variance:
        sd = out.std(axis=0, keepdims=True)
        out = out / np.maximum(sd, 1e-8)
    return out


@dataclass
class FeatureConfig:
    """Every knob of the front end, in one serialisable place.

    The acoustic model is only valid for the exact configuration it was trained
    on, so this object is stored inside the model file and checked on load.
    """

    rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    n_filters: int = 26
    n_cepstra: int = 13
    fmin: float = 20.0
    fmax: float | None = None
    preemph: float = 0.97
    lifter: int = 22
    use_energy: bool = True   # replace C0 with log frame energy
    deltas: bool = True
    delta_deltas: bool = True
    # cmvn: bool = True             # superseded by cmvn_mode
    # cmvn_variance: bool = True
    cmvn: bool = False
    cmvn_variance: bool = True
    cmvn_mode: str = "global"
    """Where cepstral normalisation statistics come from.

    "global"    - one mean/variance estimated over the training corpus and
                  stored in the model, applied identically at training and
                  decoding. The default, and the only safe choice when models
                  are trained on isolated word clips.
    "utterance" - statistics from each recording. Removes a per-recording
                  channel, but is only valid when the recording is long enough
                  to contain a representative mix of phones. On a single-word
                  clip the cepstral mean largely *is* the word identity, so
                  subtracting it destroys exactly what the model needs.
    "none"      - raw cepstra.
    """

    @property
    def frame_length(self) -> int:
        return int(round(self.frame_ms * self.rate / 1000.0))

    @property
    def hop_length(self) -> int:
        return int(round(self.hop_ms * self.rate / 1000.0))

    @property
    def n_features(self) -> int:
        n = self.n_cepstra
        if self.deltas:
            n += self.n_cepstra
        if self.delta_deltas:
            n += self.n_cepstra
        return n

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def mfcc(x: np.ndarray, config: FeatureConfig | None = None,
         stats: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Waveform -> feature matrix of shape (n_frames, config.n_features).

    `stats` is the (mean, std) pair used by cmvn_mode="global"; it is stored in
    the model file so training and decoding always share one transform.
    """
    cfg = config or FeatureConfig()
    x = np.asarray(x, dtype=np.float64)

    frames = frame_signal(preemphasis(x, cfg.preemph), cfg.frame_length, cfg.hop_length)
    if frames.shape[0] == 0:
        return np.zeros((0, cfg.n_features))
    windowed = window(frames, "hamming")

    power = power_spectrum(windowed, cfg.n_fft)
    fb = mel_filterbank(cfg.n_filters, cfg.n_fft, cfg.rate, cfg.fmin, cfg.fmax)

    # Floor before the log so silent frames give a large negative number
    # rather than -inf, which would poison every downstream statistic.
    mel_energy = np.maximum(power @ fb.T, 1e-10)
    log_mel = np.log(mel_energy)

    cepstra = dct(log_mel, type=2, axis=-1, norm="ortho")[:, : cfg.n_cepstra]
    cepstra = lifter(cepstra, cfg.lifter)

    if cfg.use_energy:
        # C0 is the mean log-mel energy, a filterbank-shaped loudness measure.
        # True frame energy is a cleaner one, so it replaces C0.
        cepstra[:, 0] = np.log(np.maximum(frame_energy(windowed), 1e-10))

    feats = [cepstra]
    if cfg.deltas:
        d1 = delta(cepstra)
        feats.append(d1)
        if cfg.delta_deltas:
            feats.append(delta(d1))
    out = np.hstack(feats)

    # if cfg.cmvn:
    #     out = cmvn(out, cfg.cmvn_variance)
    if cfg.cmvn_mode == "utterance" or cfg.cmvn:
        out = cmvn(out, cfg.cmvn_variance)
    elif cfg.cmvn_mode == "global" and stats is not None:
        out = apply_stats(out, stats)
    return out


def estimate_stats(feature_sets) -> tuple[np.ndarray, np.ndarray]:
    """Pool feature matrices and return (mean, std) for global normalisation."""
    pooled = np.vstack([np.atleast_2d(f) for f in feature_sets if np.atleast_2d(f).size])
    return pooled.mean(axis=0), np.maximum(pooled.std(axis=0), 1e-8)


def apply_stats(feat: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Apply a fixed affine normalisation estimated elsewhere."""
    mean, std = stats
    return (feat - mean) / std
