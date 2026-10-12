"""Audio input: decode anything to mono float64 at a known sample rate.

The rest of PureScribe assumes that invariant, so every path through this
module ends in the same place: one channel, one rate, samples in [-1, 1].
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

TARGET_RATE = 16000

# Telephony and most speech corpora sit at 8-16 kHz. Above 16 kHz the extra
# bandwidth carries almost no phonetic information but doubles the FFT cost,
# so we resample down rather than up.


@dataclass
class Waveform:
    """A mono signal and the rate it was sampled at."""

    samples: np.ndarray  # float64, shape (n,)
    rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.rate

    def __len__(self) -> int:
        return len(self.samples)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV with the standard library. No third-party codec needed."""
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())

    if width == 1:
        # 8-bit WAV is unsigned, biased around 128.
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, rate


def _read_ffmpeg(path: Path, rate: int) -> tuple[np.ndarray, int]:
    """Decode any container/codec by piping 16-bit PCM out of ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"cannot read {path.suffix} files without ffmpeg on PATH; "
            "convert the file to WAV first"
        )
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(path),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode(errors='replace')}")
    data = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float64) / 32768.0
    return data, rate


def load_audio(path, rate: int = TARGET_RATE, normalize: bool = True) -> Waveform:
    """Load `path` as mono at `rate`.

    WAV goes through the stdlib; anything else goes through ffmpeg, which
    already resamples for us. Resampling uses a polyphase filter so that
    downsampling cannot fold energy above Nyquist back into the speech band.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".wav":
        try:
            data, src_rate = _read_wav(path)
        except (wave.Error, ValueError):
            data, src_rate = _read_ffmpeg(path, rate)  # compressed data in a .wav container
    else:
        data, src_rate = _read_ffmpeg(path, rate)

    data = np.asarray(data, dtype=np.float64)
    if src_rate != rate:
        data = _resample(data, src_rate, rate)

    # Remove any DC offset: a constant bias inflates frame energy and skews
    # the lowest cepstral coefficients without carrying speech information.
    if data.size:
        data = data - data.mean()

    if normalize and data.size:
        peak = np.abs(data).max()
        if peak > 0:
            data = data / peak * 0.95

    return Waveform(samples=data, rate=rate)


def _resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Rational polyphase resampling: src_rate/dst_rate reduced to up/down."""
    from math import gcd

    g = gcd(int(src_rate), int(dst_rate))
    up, down = dst_rate // g, src_rate // g
    return resample_poly(data, up, down)


def frames_to_time(n_frames: int, hop_length: int, rate: int) -> np.ndarray:
    """Frame index -> seconds, measured at the centre of each frame."""
    return np.arange(n_frames) * (hop_length / rate)
