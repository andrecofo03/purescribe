"""The vectorised TokenPassingDecoder must reproduce the original search exactly.

`reference_decoder.ReferenceDecoder` is the decoder as it was before the speed-up
(dense V x V LM, per-model loops, emissions of the whole utterance). Every case
compares words, spans and score bit for bit on synthetic models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference_decoder import ReferenceDecoder  # noqa: E402

from purescribe.decoder import DecodeConfig, TokenPassingDecoder, _uniform_predecessors  # noqa: E402
from purescribe.gmm import GMM  # noqa: E402
from purescribe.hmm import HMM, LOG_ZERO  # noqa: E402
from purescribe.lexicon import SIL, Lexicon  # noqa: E402

D = 4


def make_model(rng, name, n_states, n_mix, skip=False, centre=None):
    m = HMM(n_states, D, n_mix, name=name, allow_skip=skip)
    centre = rng.normal(0, 3, D) if centre is None else centre
    for g in m.states:
        g.means = centre + rng.normal(0, 1.0, (n_mix, D))
        g.variances = rng.uniform(0.3, 2.0, (n_mix, D))
        w = rng.uniform(0.2, 1.0, n_mix)
        g.weights = w / w.sum()
    # Trained-looking transitions: random probabilities on the allowed arcs.
    allowed = m.log_trans > LOG_ZERO / 2
    for i in range(n_states):
        p = rng.uniform(0.2, 1.0, allowed[i].sum() + (i == n_states - 1))
        p /= p.sum()
        m.log_trans[i, allowed[i]] = np.log(p[: allowed[i].sum()])
        if i == n_states - 1:
            m.log_exit[i] = np.log(p[-1])
    return m


def copy_model(m, name):
    c = HMM.from_dict(m.to_dict())
    c.name = name
    return c


def build(seed, n_words=6, duplicate=True, dead_self_loop=True):
    rng = np.random.default_rng(seed)
    models = {}
    for k in range(n_words):
        models[f"w{k}"] = make_model(rng, f"w{k}", int(rng.integers(2, 6)), int(rng.integers(1, 4)))
    if duplicate:
        # Identical acoustics under two names: exact ties at every boundary.
        models["twin"] = copy_model(models["w0"], "twin")
    if dead_self_loop:
        # A state without self-loop (zero count in training), as in the real model.
        models["w1"].log_trans[0, 0] = LOG_ZERO
    models[SIL] = make_model(rng, SIL, 3, 2, skip=True, centre=np.zeros(D))
    return models, rng


def utterance(models, rng, words, frames_per_state=3, noise=0.5):
    """Frames drawn around each state's first mean, with silence at both ends."""
    seq = [SIL] + list(words) + [SIL]
    frames = []
    for w in seq:
        for g in models[w].states:
            for _ in range(frames_per_state):
                frames.append(g.means[0] + rng.normal(0, noise, D))
    return np.array(frames)


def assert_same(models, lexicon, config, X):
    ref = ReferenceDecoder(models, lexicon, config).decode(X)
    new = TokenPassingDecoder(models, lexicon, config).decode(X)
    assert new.words == ref.words
    assert [(s.word, s.start_frame, s.end_frame) for s in new.spans] == \
           [(s.word, s.start_frame, s.end_frame) for s in ref.spans]
    assert new.score == ref.score  # bit for bit, not approximately
    assert new.n_frames == ref.n_frames
    return ref


CONFIGS = [
    DecodeConfig(),
    DecodeConfig(beam=float("inf")),
    DecodeConfig(beam=15.0),
    DecodeConfig(beam=float("inf"), max_active=12),
    DecodeConfig(lm_scale=10.0, word_penalty=-6.0),
    DecodeConfig(lm_scale=1.0, word_penalty=3.0, beam=40.0),
]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"lm{c.lm_scale}_wp{c.word_penalty}_b{c.beam}_m{c.max_active}")
def test_uniform_lexicon_matches_reference(seed, config):
    models, rng = build(seed)
    lexicon = Lexicon([w for w in models if w != SIL])
    words = list(rng.choice([w for w in models if w != SIL], size=4))
    X = utterance(models, rng, words)
    ref = assert_same(models, lexicon, config, X)
    assert ref.n_frames == X.shape[0]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("config", CONFIGS[:3] + CONFIGS[4:5], ids=lambda c: f"lm{c.lm_scale}_wp{c.word_penalty}_b{c.beam}")
def test_bigram_lexicon_matches_reference(seed, config):
    models, rng = build(seed)
    vocab = [w for w in models if w != SIL]
    sentences = [list(rng.choice(vocab, size=int(rng.integers(1, 5)))) for _ in range(20)]
    lexicon = Lexicon(vocab).fit(sentences)
    assert lexicon.bigram
    decoder = TokenPassingDecoder(models, lexicon, config)
    assert not decoder.uniform
    X = utterance(models, rng, sentences[0])
    assert_same(models, lexicon, config, X)


def test_silence_only_and_single_frame():
    models, rng = build(11)
    lexicon = Lexicon([w for w in models if w != SIL])
    silence = np.array([g.means[0] for g in models[SIL].states for _ in range(4)])
    ref = assert_same(models, lexicon, DecodeConfig(), silence)
    assert any(s.word == SIL for s in ref.spans)
    assert_same(models, lexicon, DecodeConfig(), silence[:1])
    assert TokenPassingDecoder(models, lexicon).decode(np.zeros((0, D))).words == []


def test_unreachable_end_returns_empty_like_reference():
    models, rng = build(12)
    lexicon = Lexicon([w for w in models if w != SIL])
    # Fewer frames than any model's minimum duration (silence can skip, so drop it).
    models = {w: m for w, m in models.items() if w != SIL}
    lexicon = Lexicon(list(models), silence="<none>")
    X = rng.normal(0, 1, (1, D))
    for m in models.values():
        m.log_exit[:] = LOG_ZERO
    assert_same(models, lexicon, DecodeConfig(), X)


def test_uniform_decoder_builds_no_dense_lm():
    models, _ = build(13)
    decoder = TokenPassingDecoder(models, Lexicon([w for w in models if w != SIL]))
    assert decoder.uniform and decoder._lm is None
    assert decoder._lm_successor.shape == (len(decoder.names),)


def test_stacked_emissions_are_bitwise_equal_to_per_model():
    models, rng = build(14)
    decoder = TokenPassingDecoder(models, Lexicon([w for w in models if w != SIL]))
    X = rng.normal(0, 2, (7, D))
    full = decoder.log_emissions(X)
    states = np.arange(decoder.n_total)
    for t in range(X.shape[0]):
        assert np.array_equal(decoder._emissions_at(X[t], states), full[t])
    some = states[::3]
    partial = decoder._emissions_at(X[0], some)
    assert np.array_equal(partial[some], full[0, some])
    assert not partial[np.setdiff1d(states, some)].any()


def test_threaded_emission_chunks_change_nothing(monkeypatch):
    import purescribe.decoder as decoder_module

    models, rng = build(16)
    lexicon = Lexicon([w for w in models if w != SIL])
    X = utterance(models, rng, ["w2", "w0", "w3"])
    serial = TokenPassingDecoder(models, lexicon).decode(X)
    monkeypatch.setattr(decoder_module, "EMISSION_CHUNK", 3)  # many chunks -> thread pool
    threaded_decoder = TokenPassingDecoder(models, lexicon)
    threaded = threaded_decoder.decode(X)
    assert threaded_decoder._pool is not None
    assert threaded.words == serial.words and threaded.score == serial.score
    full = threaded_decoder.log_emissions(X[:2])
    assert np.array_equal(threaded_decoder._emissions_at(X[1], np.arange(threaded_decoder.n_total)), full[1])


def test_uniform_predecessor_follows_dense_argmax_on_rounding_ties():
    rng = np.random.default_rng(15)
    for _ in range(2000):
        top = -float(rng.uniform(10, 5000))
        exits = np.full(8, LOG_ZERO)
        # Exits within a few ulps of each other: their sums with lm can round to ties.
        exits[rng.choice(8, 4, replace=False)] = top - rng.integers(0, 4, 4) * np.spacing(top)
        lm = -rng.uniform(0, 20, 5)
        wp = float(rng.choice([0.0, -6.0, 2.5]))
        dense = exits[:, None] + lm[None, :] + wp
        v_best = int(exits.argmax())
        got = _uniform_predecessors(exits, v_best, lm, wp)
        assert np.array_equal(got, dense.argmax(axis=0))
