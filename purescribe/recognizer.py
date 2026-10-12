"""The assembled recogniser: features + acoustic models + lexicon + search.

This is the object a user holds. It owns the FeatureConfig its models were
trained on, so audio can never be parameterised one way and scored another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import serialize
from .audio import Waveform, load_audio
from .decoder import DecodeConfig, DecodeResult, TokenPassingDecoder, decode_isolated
from .features import FeatureConfig, mfcc
from .hmm import HMM
from .lexicon import SIL, Lexicon
from .vad import VADConfig, detect_speech

FORMAT_VERSION = 1


@dataclass
class Word:
    text: str
    start: float
    end: float
    score: float = 0.0


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)
    duration: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def to_srt(self, max_words: int = 8) -> str:
        """Group words into caption cues and format as SubRip."""
        lines, cue = [], []
        out, index = [], 1
        for w in self.words:
            cue.append(w)
            if len(cue) >= max_words:
                out.append(_srt_cue(index, cue)); index += 1; cue = []
        if cue:
            out.append(_srt_cue(index, cue))
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "duration": self.duration,
            "words": [{"text": w.text, "start": round(w.start, 3),
                       "end": round(w.end, 3), "score": round(w.score, 3)}
                      for w in self.words],
        }


def _srt_cue(index: int, words: list[Word]) -> str:
    def ts(t):
        # Round to milliseconds *first*. Rounding the fractional part on its own
        # lets 1.9996 s render as "01,1000" - a four-digit field no SRT parser
        # accepts.
        total_ms = int(round(max(t, 0.0) * 1000))
        h, rem = divmod(total_ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        sec, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
    return (f"{index}\n{ts(words[0].start)} --> {ts(words[-1].end)}\n"
            f"{' '.join(w.text for w in words)}\n")


class Recognizer:
    """Holds everything needed to turn audio into text."""

    def __init__(
        self,
        models: dict[str, HMM],
        lexicon: Lexicon,
        feature_config: FeatureConfig | None = None,
        decode_config: DecodeConfig | None = None,
        vad_config: VADConfig | None = None,
        feature_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.models = models
        self.lexicon = lexicon
        # (mean, std) for cmvn_mode="global". Stored with the model so the
        # exact transform used in training is reproduced at decode time.
        self.feature_stats = feature_stats
        self.feature_config = feature_config or FeatureConfig()
        self.decode_config = decode_config or DecodeConfig()
        self.vad_config = vad_config or VADConfig(
            frame_ms=self.feature_config.frame_ms, hop_ms=self.feature_config.hop_ms
        )
        self._decoder: TokenPassingDecoder | None = None

    @property
    def decoder(self) -> TokenPassingDecoder:
        if self._decoder is None:
            self._decoder = TokenPassingDecoder(self.models, self.lexicon, self.decode_config)
        return self._decoder

    @property
    def vocabulary(self) -> list[str]:
        return list(self.lexicon.words)

    # -- recognition -----------------------------------------------------

    def features(self, x: np.ndarray) -> np.ndarray:
        # return mfcc(x, self.feature_config)
        return mfcc(x, self.feature_config, stats=self.feature_stats)

    #: Recordings longer than this are split on silence before decoding.
    vad_threshold_s: float = 30.0

    def transcribe(self, source, use_vad: bool | None = None) -> Transcript:
        """Transcribe a file path, a Waveform, or a raw sample array.

        `use_vad=None` (the default) decides by duration. Short recordings are
        decoded whole, because the search already has a silence model in the
        loop and decides word boundaries globally; pre-segmenting them measured
        strictly worse - on a connected-digit test it cut words at boundaries
        and drove word error from 0.0% to 4.9%, all of it deletions. VAD earns
        its place only on long recordings, where it bounds the cost of a search
        that is linear in frames but must keep every word active at once.
        """
        wave = self._as_waveform(source)
        hop_s = self.feature_config.hop_length / self.feature_config.rate

        if use_vad is None:
            use_vad = wave.duration > self.vad_threshold_s

        if not use_vad:
            result = self.decoder.decode(self.features(wave.samples))
            return self._to_transcript(result, 0.0, hop_s, wave.duration)

        # Features are extracted once over the whole recording so that cepstral
        # normalisation sees the channel, not each segment's speech-to-silence
        # ratio. Normalising per segment was a bug: a short segment padded with
        # silence gets a different cepstral mean from the trimmed clips the
        # models were trained on, and every word model is then mismatched.
        feats = self.features(wave.samples)
        segments = detect_speech(wave.samples, wave.rate, self.vad_config)
        words: list[Word] = []
        for seg in segments:
            # lo = int(seg.start * wave.rate)
            # hi = min(len(wave.samples), int(seg.end * wave.rate))
            # chunk = wave.samples[lo:hi]
            # if chunk.size < self.feature_config.frame_length:
            #     continue
            # result = self.decoder.decode(self.features(chunk))
            lo = max(0, seg.start_frame)
            hi = min(feats.shape[0], seg.end_frame)
            if hi - lo < 2:
                continue
            result = self.decoder.decode(feats[lo:hi])
            words.extend(self._to_transcript(result, lo * hop_s, hop_s, 0.0).words)
        return Transcript(words=words, duration=wave.duration)

    def classify(self, source) -> tuple[str, float]:
        """Whole-clip single-word classification, bypassing the search."""
        wave = self._as_waveform(source)
        return decode_isolated(self.models, self.features(wave.samples), exclude=(self.lexicon.silence,))

    def _to_transcript(self, result: DecodeResult, offset: float,
                       hop_s: float, duration: float) -> Transcript:
        words = [
            Word(text=s.word,
                 start=offset + s.start_frame * hop_s,
                 end=offset + (s.end_frame + 1) * hop_s,
                 score=result.score)
            for s in result.spans
            if s.word != self.lexicon.silence
        ]
        return Transcript(words=words, duration=duration or (offset + result.n_frames * hop_s))

    def _as_waveform(self, source) -> Waveform:
        if isinstance(source, Waveform):
            if source.rate != self.feature_config.rate:
                raise ValueError(
                    f"waveform is {source.rate} Hz but models expect "
                    f"{self.feature_config.rate} Hz"
                )
            return source
        if isinstance(source, np.ndarray):
            return Waveform(np.asarray(source, dtype=np.float64), self.feature_config.rate)
        return load_audio(source, rate=self.feature_config.rate)

    # -- persistence -----------------------------------------------------

    def save(self, path) -> Path:
        return serialize.save(path, {
            "format": FORMAT_VERSION,
            "feature_config": self.feature_config.to_dict(),
            "decode_config": dict(self.decode_config.__dict__),
            "lexicon": self.lexicon.to_dict(),
            "feature_stats": ({"mean": self.feature_stats[0], "std": self.feature_stats[1]}
                              if self.feature_stats is not None else {}),
            "models": {name: m.to_dict() for name, m in self.models.items()},
        })

    @classmethod
    def load(cls, path) -> "Recognizer":
        d = serialize.load(path)
        if int(d.get("format", 0)) != FORMAT_VERSION:
            raise ValueError(f"unsupported model format: {d.get('format')}")
        stats = d.get("feature_stats") or {}
        return cls(
            models={n: HMM.from_dict(m) for n, m in d["models"].items()},
            lexicon=Lexicon.from_dict(d["lexicon"]),
            feature_config=FeatureConfig.from_dict(d["feature_config"]),
            decode_config=DecodeConfig(**d["decode_config"]),
            feature_stats=((stats["mean"], stats["std"]) if "mean" in stats else None),
        )

    def __repr__(self) -> str:
        return (f"Recognizer(vocab={len(self.lexicon)}, "
                f"models={len(self.models)}, dim={self.feature_config.n_features})")
