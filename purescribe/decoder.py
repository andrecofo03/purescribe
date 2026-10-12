"""Connected-word decoding by token passing (Young, Russell & Thornton 1989).

A single Viterbi search runs over every word model simultaneously. The models
are wired into a loop: a token that reaches a model's exit may re-enter any
model, paying the language-model score for that transition. Because the search
is one pass over a composite state space, word boundaries are decided globally
rather than greedily - which is the whole reason connected-word recognition
works at all.

Two knobs matter in practice and both are exposed:

  lm_scale   Acoustic likelihoods are computed over 39 correlated dimensions
             and are wildly over-confident, so the LM must be scaled up to
             have any influence. 8-12 is typical.
  word_penalty  A constant added per word. Negative values suppress insertions;
             positive values suppress deletions. It must be read together with
             lm_scale: a uniform LM over V words contributes a constant
             lm_scale*log(1/(V+1)) at every word entry, so raising lm_scale
             silently raises the cost of emitting any word at all. With the
             defaults below and a 10-word uniform vocabulary the net entry cost
             is about +2.6, which sits in the middle of the measured plateau
             where digit strings decode without deletions or insertions. Fit a
             real bigram and lm_scale of 8-12 becomes worthwhile, but
             word_penalty must then rise with it.

The one setting to avoid is a *net positive* entry cost,
lm_scale*log(1/(V+1)) + word_penalty > 0. That pays the search for crossing a
word boundary, and it will happily re-enter the background model over and over
to collect the bonus, emitting a transcript of pure silence. With the defaults
the net cost stays negative for any vocabulary size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hmm import HMM, LOG_ZERO
from .lexicon import EOS, Lexicon


@dataclass
class DecodeConfig:
    # lm_scale: float = 10.0
    # word_penalty: float = -6.0
    # word_penalty: float = 5.0       # net entry reward; see the note below
    lm_scale: float = 1.0
    word_penalty: float = 0.0
    beam: float = 250.0          # log-likelihood width; inf disables pruning
    max_active: int = 0          # hard cap on surviving states, 0 = unlimited


@dataclass
class WordSpan:
    word: str
    start_frame: int
    end_frame: int
    start: float = 0.0
    end: float = 0.0


@dataclass
class DecodeResult:
    words: list[str] = field(default_factory=list)
    spans: list[WordSpan] = field(default_factory=list)
    score: float = LOG_ZERO
    n_frames: int = 0

    @property
    def text(self) -> str:
        return " ".join(self.words)


class TokenPassingDecoder:
    """One-pass Viterbi beam search across a loop of word HMMs."""

    def __init__(self, models: dict[str, HMM], lexicon: Lexicon,
                 config: DecodeConfig | None = None):
        self.lexicon = lexicon
        self.config = config or DecodeConfig()

        self.names = [w for w in lexicon.all_models if w in models]
        if not self.names:
            raise ValueError("no acoustic models match the lexicon")
        self.models = {w: models[w] for w in self.names}

        # Flatten every model's states into one global index space.
        self.offsets: dict[str, int] = {}
        total = 0
        for w in self.names:
            self.offsets[w] = total
            total += self.models[w].n_states
        self.n_total = total

        # Precompute the LM matrix once: rows are predecessors, columns
        # successors. Saves a dictionary lookup per word pair per frame.
        self._lm = np.array(
            [[lexicon.log_prob(w, v) for w in self.names] for v in self.names],
            dtype=np.float64,
        )
        self._lm_init = np.array(
            [lexicon.log_prob(w, None) for w in self.names], dtype=np.float64
        )
        self._lm_final = np.array(
            [lexicon.log_prob(EOS, v) if lexicon.bigram else 0.0 for v in self.names],
            dtype=np.float64,
        )

    # -- emissions -------------------------------------------------------

    def log_emissions(self, X: np.ndarray) -> np.ndarray:
        """Stack every model's frame likelihoods -> (T, n_total)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.empty((X.shape[0], self.n_total))
        for w in self.names:
            off = self.offsets[w]
            out[:, off : off + self.models[w].n_states] = self.models[w].log_emission(X)
        return out

    # -- search ----------------------------------------------------------

    def decode(self, X: np.ndarray) -> DecodeResult:
        """Return the most likely word sequence for feature matrix X."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        T = X.shape[0]
        if T == 0:
            return DecodeResult()

        cfg = self.config
        logB = self.log_emissions(X)
        V = len(self.names)

        delta = np.full(self.n_total, LOG_ZERO)
        bp = np.full(self.n_total, -1, dtype=np.int64)   # -> word link record

        # Word link records: (word index, frame it ended on, parent record).
        # The decoded transcript is read out by walking this chain backwards.
        links: list[tuple[int, int, int]] = []

        # t = 0: enter every model from the sentence-start context.
        for wi, w in enumerate(self.names):
            m, off = self.models[w], self.offsets[w]
            entry = m.log_entry + cfg.lm_scale * self._lm_init[wi] + cfg.word_penalty
            delta[off : off + m.n_states] = entry + logB[0, off : off + m.n_states]

        for t in range(1, T):
            new_delta = np.full(self.n_total, LOG_ZERO)
            new_bp = np.full(self.n_total, -1, dtype=np.int64)

            # 1. Propagate inside each model (ordinary Viterbi recursion).
            for w in self.names:
                m, off = self.models[w], self.offsets[w]
                sl = slice(off, off + m.n_states)
                scores = delta[sl][:, None] + m.log_trans        # (from, to)
                best_from = scores.argmax(axis=0)
                new_delta[sl] = scores.max(axis=0) + logB[t, sl]
                new_bp[sl] = bp[sl][best_from]

            # 2. Score exiting each model at t-1.
            exit_score = np.full(V, LOG_ZERO)
            exit_bp = np.full(V, -1, dtype=np.int64)
            for wi, w in enumerate(self.names):
                m, off = self.models[w], self.offsets[w]
                sl = slice(off, off + m.n_states)
                cand = delta[sl] + m.log_exit
                k = int(cand.argmax())
                exit_score[wi] = cand[k]
                exit_bp[wi] = bp[sl][k]

            # 3. Cross word boundaries. Entry into w takes the best predecessor
            #    v, which is exact for a bigram because the LM score is applied
            #    at the boundary and the predecessor identity is recorded.
            transition = exit_score[:, None] + cfg.lm_scale * self._lm + cfg.word_penalty
            best_prev = transition.argmax(axis=0)                # (V,) over v
            best_score = transition.max(axis=0)

            link_cache: dict[int, int] = {}
            for wi, w in enumerate(self.names):
                if best_score[wi] <= LOG_ZERO / 2:
                    continue
                m, off = self.models[w], self.offsets[w]
                start = best_score[wi] + m.log_entry[0] + logB[t, off]
                if start <= new_delta[off]:
                    continue  # staying inside the current word scores better
                vi = int(best_prev[wi])
                if vi not in link_cache:
                    link_cache[vi] = len(links)
                    links.append((vi, t - 1, int(exit_bp[vi])))
                new_delta[off] = start
                new_bp[off] = link_cache[vi]

            delta, bp = new_delta, new_bp
            delta = self._prune(delta)

        return self._traceback(delta, bp, links, T)

    def _prune(self, delta: np.ndarray) -> np.ndarray:
        """Beam pruning: drop states far behind the current best hypothesis."""
        cfg = self.config
        best = delta.max()
        if not np.isfinite(best) or best <= LOG_ZERO / 2:
            return delta
        if np.isfinite(cfg.beam):
            delta = np.where(delta < best - cfg.beam, LOG_ZERO, delta)
        if cfg.max_active and np.count_nonzero(delta > LOG_ZERO / 2) > cfg.max_active:
            cutoff = np.partition(delta, -cfg.max_active)[-cfg.max_active]
            delta = np.where(delta < cutoff, LOG_ZERO, delta)
        return delta

    def _traceback(self, delta, bp, links, T) -> DecodeResult:
        """Walk the word link records back from the best final state."""
        final_score = np.full(len(self.names), LOG_ZERO)
        final_bp = np.full(len(self.names), -1, dtype=np.int64)
        for wi, w in enumerate(self.names):
            m, off = self.models[w], self.offsets[w]
            sl = slice(off, off + m.n_states)
            cand = delta[sl] + m.log_exit + self.config.lm_scale * self._lm_final[wi]
            k = int(cand.argmax())
            final_score[wi] = cand[k]
            final_bp[wi] = bp[sl][k]

        best_w = int(final_score.argmax())
        if final_score[best_w] <= LOG_ZERO / 2:
            return DecodeResult(n_frames=T)

        # Close the chain with the final word, then unwind it.
        chain: list[tuple[int, int]] = [(best_w, T - 1)]
        cursor = int(final_bp[best_w])
        while cursor >= 0:
            wi, end, parent = links[cursor]
            chain.append((wi, end))
            cursor = parent
        chain.reverse()

        spans, start = [], 0
        for wi, end in chain:
            spans.append(WordSpan(self.names[wi], start, end))
            start = end + 1

        silence = self.lexicon.silence
        return DecodeResult(
            words=[s.word for s in spans if s.word != silence],
            spans=spans,
            score=float(final_score[best_w]),
            n_frames=T,
        )


def decode_isolated(models: dict[str, HMM], X: np.ndarray, exclude=()) -> tuple[str, float]:
    """Whole-utterance classification: score X against each model, take the best.

    Correct only when the clip holds exactly one word. Useful as a baseline and
    for validating acoustic models independently of the search.
    """
    best, best_score = "", LOG_ZERO
    for name, model in models.items():
        if name in exclude:
            continue
        s = model.score(X)
        if s > best_score:
            best, best_score = name, s
    return best, best_score
