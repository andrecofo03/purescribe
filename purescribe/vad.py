"""Voice activity detection by short-time energy and zero-crossing rate.

The classic Rabiner-Sambur endpointer, which needs no training. Energy alone
misses unvoiced fricatives - /s/ and /f/ carry little power but cross zero
constantly - so the two measures are combined, and hysteresis stops a decision
flapping frame to frame in the middle of a word.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp import frame_energy, frame_signal, zero_crossing_rate


@dataclass
class VADConfig:
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    noise_ms: float = 200.0      # leading audio assumed to be background
    energy_factor: float = 4.0   # speech onset, relative to noise floor
    release_factor: float = 2.0  # lower threshold for staying in speech
    zcr_factor: float = 1.6
    min_speech_ms: float = 120.0
    min_silence_ms: float = 180.0
    pad_ms: float = 60.0         # keep context either side of each segment


@dataclass
class Segment:
    start: float          # seconds
    end: float
    start_frame: int
    end_frame: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_speech(
    samples: np.ndarray,
    rate: int,
    config: VADConfig | None = None,
) -> list[Segment]:
    """Return the speech regions of `samples` as time segments."""
    cfg = config or VADConfig()
    frame_len = int(round(cfg.frame_ms * rate / 1000.0))
    hop = int(round(cfg.hop_ms * rate / 1000.0))

    frames = frame_signal(np.asarray(samples, dtype=np.float64), frame_len, hop)
    if frames.shape[0] == 0:
        return []

    energy = frame_energy(frames)
    zcr = zero_crossing_rate(frames)

    # Estimate the noise floor from the opening frames, falling back to the
    # signal's own low percentile when the recording starts mid-speech.
    n_noise = max(1, int(cfg.noise_ms / cfg.hop_ms))
    n_noise = min(n_noise, frames.shape[0])
    noise_e = float(np.median(energy[:n_noise]))
    noise_z = float(np.median(zcr[:n_noise]))
    floor_e = float(np.percentile(energy, 10))
    noise_e = max(min(noise_e, floor_e * 4.0), 1e-12)

    on_threshold = noise_e * cfg.energy_factor
    off_threshold = noise_e * cfg.release_factor
    zcr_threshold = min(noise_z * cfg.zcr_factor, 0.5)

    # Hysteresis (Schmitt trigger): enter speech on the high threshold, leave
    # only below the low one, so a brief dip inside a word does not split it.
    active = np.zeros(frames.shape[0], dtype=bool)
    state = False
    for i in range(frames.shape[0]):
        if state:
            state = energy[i] > off_threshold or zcr[i] > zcr_threshold
        else:
            state = energy[i] > on_threshold
        active[i] = state

    segments = _runs_to_segments(active, cfg, hop, rate, frames.shape[0])
    return segments


def _runs_to_segments(active, cfg, hop, rate, n_frames) -> list[Segment]:
    """Turn a boolean activity mask into padded, gap-merged segments."""
    min_speech = max(1, int(cfg.min_speech_ms / cfg.hop_ms))
    min_silence = max(1, int(cfg.min_silence_ms / cfg.hop_ms))
    pad = int(cfg.pad_ms / cfg.hop_ms)

    # Collect raw runs of activity.
    runs: list[list[int]] = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, n_frames])

    # Bridge short pauses: a gap shorter than min_silence is inside a word,
    # not between two utterances.
    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] < min_silence:
            merged[-1][1] = run[1]
        else:
            merged.append(list(run))

    out = []
    for lo, hi in merged:
        if hi - lo < min_speech:
            continue
        lo = max(0, lo - pad)
        hi = min(n_frames, hi + pad)
        out.append(Segment(
            start=lo * hop / rate,
            end=hi * hop / rate,
            start_frame=lo,
            end_frame=hi,
        ))
    return out


def trim_silence(samples: np.ndarray, rate: int, config: VADConfig | None = None) -> np.ndarray:
    """Crop to the first through last detected speech frame.

    Used on training clips so that word models learn the word, not the pause
    around it.
    """
    segs = detect_speech(samples, rate, config)
    if not segs:
        return np.asarray(samples, dtype=np.float64)
    lo = int(segs[0].start * rate)
    hi = min(len(samples), int(segs[-1].end * rate))
    return np.asarray(samples[lo:hi], dtype=np.float64)
