"""Training: labelled audio -> a Recognizer.

Two stages, in the order HTK made standard.

1. Isolated bootstrap. Each word model is flat-started on clips of that word
   alone and refined by segmental k-means. Cheap, stable, and it needs no
   alignment to exist beforehand.
2. Embedded re-estimation. Whole utterances are aligned against a composite
   model built by concatenating the words of their transcript. Frames are then
   re-assigned across word boundaries, so boundary states learn from real
   connected speech instead of from clipped training tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import load_audio
from .decoder import DecodeConfig
from .features import FeatureConfig, apply_stats, estimate_stats, mfcc
from .hmm import HMM, LOG_ZERO
from .lexicon import SIL, Lexicon
from .recognizer import Recognizer
from .vad import VADConfig, detect_speech, trim_silence

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aiff", ".wma"}


@dataclass
class TrainConfig:
    n_mixtures: int = 2
    frames_per_state: float = 4.0   # target dwell time per state
    min_states: int = 3
    max_states: int = 24
    # sil_states: int = 3
    sil_states: int = 1
    """States in the background model. One, deliberately.

    A multi-state silence model has a left-to-right chain, and a chain can
    follow a *trajectory* - which lets it track speech through its states and
    explain a whole word as background. Measured on a four-word corpus this was
    the difference between 66.7% and 0.0% word error. Silence has no temporal
    structure to model, so it gets one state and earns its flexibility from
    mixture components instead.
    """
    iterations: int = 10
    embedded_iterations: int = 4
    method: str = "viterbi"
    trim: bool = True               # endpoint training clips before use
    seed: int = 0
    var_floor: float = 0.001
    """Variance floor, as a fraction of the global feature variance.

    A 39-dimensional diagonal Gaussian needs 78 free parameters. With a small
    corpus a state can easily own fewer frames than that, and its variances
    collapse towards zero - the model becomes over-confident, punishes any
    unseen deviation enormously, and loses to the broad silence model at decode
    time. Flooring at a few percent of the global variance is what keeps a
    sparsely-trained state honest.
    """


# -- corpus loading ------------------------------------------------------

def load_word_corpus(root, feature_config: FeatureConfig, config: TrainConfig):
    """Read `root/<word>/*.wav` into {word: [feature matrices]}.

    One directory per vocabulary word is the simplest labelling scheme that
    exists, and for a constrained vocabulary it is usually the right one.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    corpus: dict[str, list[np.ndarray]] = {}
    for word_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        clips = sorted(p for p in word_dir.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
        if not clips:
            continue
        feats = []
        for clip in clips:
            wave = load_audio(clip, rate=feature_config.rate)
            samples = trim_silence(wave.samples, wave.rate) if config.trim else wave.samples
            if samples.size < feature_config.frame_length:
                continue
            f = mfcc(samples, feature_config)
            if f.shape[0] >= config.min_states:
                feats.append(f)
        if feats:
            corpus[word_dir.name] = feats
    if not corpus:
        raise ValueError(f"no usable audio found under {root}")
    return corpus


def load_manifest(path, feature_config: FeatureConfig):
    """Read a JSONL manifest of {"audio": ..., "text": ...} for embedded training."""
    path = Path(path)
    items = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            audio = Path(rec["audio"])
            if not audio.is_absolute():
                audio = path.parent / audio
            wave = load_audio(audio, rate=feature_config.rate)
            items.append((mfcc(wave.samples, feature_config), str(rec["text"]).split()))
    return items


def collect_silence(root, feature_config: FeatureConfig, max_clips: int = 200):
    """Harvest non-speech frames to train the background model.

    Uses whatever the VAD rejects. A silence model trained on the same rooms
    and microphones as the speech is far more useful than a generic one.
    """
    root = Path(root)
    vad_cfg = VADConfig(frame_ms=feature_config.frame_ms, hop_ms=feature_config.hop_ms)
    hop_s = feature_config.hop_length / feature_config.rate
    out: list[np.ndarray] = []

    clips = [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES]
    for clip in clips[:max_clips]:
        wave = load_audio(clip, rate=feature_config.rate)
        feats = mfcc(wave.samples, feature_config)
        if feats.shape[0] < 6:
            continue
        speech = detect_speech(wave.samples, wave.rate, vad_cfg)
        mask = np.ones(feats.shape[0], dtype=bool)
        for seg in speech:
            lo = max(0, int(seg.start / hop_s))
            hi = min(feats.shape[0], int(seg.end / hop_s) + 1)
            mask[lo:hi] = False
        quiet = feats[mask]
        if quiet.shape[0] >= 3:
            out.append(quiet)
    return out


# -- model construction --------------------------------------------------

def choose_n_states(feats: list[np.ndarray], config: TrainConfig) -> int:
    """Pick the state count from the median duration of the training tokens.

    A left-to-right model must emit at least one frame per state, so a model
    with more states than the shortest token can never align. Sizing from the
    data keeps duration modelling sane for both short and long words.
    """
    lengths = [f.shape[0] for f in feats]
    median = float(np.median(lengths))
    n = int(round(median / config.frames_per_state))
    n = max(config.min_states, min(config.max_states, n))
    return max(1, min(n, min(lengths)))  # never exceed the shortest token


def train(
    corpus: dict[str, list[np.ndarray]],
    feature_config: FeatureConfig | None = None,
    config: TrainConfig | None = None,
    silence_feats: list[np.ndarray] | None = None,
    transcripts: list[list[str]] | None = None,
    verbose: bool = True,
) -> Recognizer:
    """Train a full recogniser from {word: [feature matrices]}."""
    fcfg = feature_config or FeatureConfig()
    tcfg = config or TrainConfig()
    dim = fcfg.n_features

    # Global normalisation: one transform estimated over the whole corpus and
    # carried in the model, so decoding sees features on exactly the training
    # scale. Estimated before any model is fitted.
    stats = None
    if fcfg.cmvn_mode == "global":
        stats = estimate_stats([f for feats in corpus.values() for f in feats]
                               + list(silence_feats or []))
        corpus = {w: [apply_stats(f, stats) for f in feats] for w, feats in corpus.items()}
        if silence_feats:
            silence_feats = [apply_stats(f, stats) for f in silence_feats]

    models: dict[str, HMM] = {}
    for word in sorted(corpus):
        feats = corpus[word]
        n_states = choose_n_states(feats, tcfg)
        model = HMM(n_states, dim, tcfg.n_mixtures, name=word,
                    var_floor=tcfg.var_floor)
        model.fit(feats, n_iter=tcfg.iterations, method=tcfg.method, seed=tcfg.seed)
        models[word] = model
        if verbose:
            print(f"  {word:<20s} {len(feats):4d} clips  {n_states:2d} states  "
                  f"{tcfg.n_mixtures} mix")

    # The silence model is flat: background noise has no temporal structure
    # worth modelling, so a single well-fitted state beats a chain. It still
    # needs at least two mixture components - one Gaussian over a mixture of
    # room tone and breath ends up with an enormous variance and swallows
    # everything.
    if silence_feats:
        sil = HMM(tcfg.sil_states, dim, max(tcfg.n_mixtures, 2), name=SIL,
                  allow_skip=True, var_floor=tcfg.var_floor)
        sil.fit(silence_feats, n_iter=tcfg.iterations, method=tcfg.method, seed=tcfg.seed)
    else:
        sil = HMM(1, dim, 1, name=SIL)
        pooled = np.vstack([f for feats in corpus.values() for f in feats])
        sil.states[0].set_floor_from(pooled)
        sil.states[0].fit(pooled, n_iter=5)
        # Widen it so silence is a weak, non-committal explanation for any
        # frame rather than a competitor to the word models.
        sil.states[0].variances *= 4.0
    models[SIL] = sil
    if verbose:
        print(f"  {SIL:<20s} {sil.n_states:2d} states")

    lexicon = Lexicon(sorted(corpus))
    if transcripts:
        lexicon.fit(transcripts)

    return Recognizer(models, lexicon, fcfg, DecodeConfig(), feature_stats=stats)


# -- embedded re-estimation ----------------------------------------------

def _composite(models: dict[str, HMM], words: list[str], optional_silence: str | None):
    """Concatenate word models into one HMM matching a known transcript.

    Returns the composite plus a map from composite state index back to
    (word, local state), which is how aligned frames find their way home.
    """
    chain: list[str] = []
    for i, w in enumerate(words):
        if optional_silence and i == 0:
            chain.append(optional_silence)
        chain.append(w)
        if optional_silence:
            chain.append(optional_silence)

    chain = [w for w in chain if w in models]
    if not chain:
        return None, []

    sizes = [models[w].n_states for w in chain]
    total = sum(sizes)
    dim = models[chain[0]].n_features

    comp = HMM(total, dim, 1, name="+".join(chain), allow_skip=True)
    comp.log_trans = np.full((total, total), LOG_ZERO)
    comp.log_entry = np.full(total, LOG_ZERO)
    comp.log_exit = np.full(total, LOG_ZERO)

    index: list[tuple[str, int]] = []
    offset = 0
    for ci, w in enumerate(chain):
        m = models[w]
        n = m.n_states
        comp.states[offset : offset + n] = m.states
        comp.log_trans[offset : offset + n, offset : offset + n] = m.log_trans
        index.extend((w, i) for i in range(n))

        if ci == 0:
            comp.log_entry[offset : offset + n] = m.log_entry
        if ci + 1 < len(chain):
            # Exit of this model feeds the entry of the next one.
            nxt = models[chain[ci + 1]]
            nxt_off = offset + n
            for i in range(n):
                if m.log_exit[i] > LOG_ZERO / 2:
                    for j in range(nxt.n_states):
                        if nxt.log_entry[j] > LOG_ZERO / 2:
                            comp.log_trans[offset + i, nxt_off + j] = (
                                m.log_exit[i] + nxt.log_entry[j]
                            )
        else:
            comp.log_exit[offset : offset + n] = m.log_exit
        offset += n

    return comp, index


def embedded_reestimate(
    recognizer: Recognizer,
    items: list[tuple[np.ndarray, list[str]]],
    n_iter: int = 4,
    optional_silence: str | None = SIL,
    verbose: bool = True,
) -> Recognizer:
    """Refine every model jointly on full utterances with known transcripts."""
    models = recognizer.models
    for it in range(n_iter):
        pooled: dict[str, list[list[np.ndarray]]] = {
            name: [[] for _ in range(m.n_states)] for name, m in models.items()
        }
        total_ll, used = 0.0, 0

        for X, words in items:
            comp, index = _composite(models, words, optional_silence)
            if comp is None or X.shape[0] < comp.n_states:
                continue  # transcript cannot fit in the available frames
            path, ll = comp.viterbi(comp.log_emission(X))
            total_ll += ll
            used += 1
            for t, s in enumerate(path):
                word, local = index[s]
                pooled[word][local].append(X[t])

        if not used:
            break

        for name, per_state in pooled.items():
            model = models[name]
            for i, frames in enumerate(per_state):
                if len(frames) >= model.n_mixtures:
                    model.states[i].fit(np.vstack(frames), n_iter=8)
        if verbose:
            print(f"  embedded pass {it + 1}: {used} utterances, "
                  f"mean log-lik {total_ll / used:8.1f}")
    recognizer._decoder = None  # models changed; force the decoder to rebuild
    return recognizer


def train_from_directory(root, feature_config=None, config=None, verbose=True) -> Recognizer:
    """Convenience path: `root/<word>/*.wav` -> trained Recognizer."""
    fcfg = feature_config or FeatureConfig()
    tcfg = config or TrainConfig()
    if verbose:
        print(f"reading corpus from {root}")
    corpus = load_word_corpus(root, fcfg, tcfg)
    silence = collect_silence(root, fcfg)
    if verbose:
        print(f"training {len(corpus)} word models "
              f"({sum(len(v) for v in corpus.values())} clips, "
              f"{len(silence)} silence regions)")
    return train(corpus, fcfg, tcfg, silence_feats=silence, verbose=verbose)
