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

Cost. With thousands of words the search is vectorised over the global state
space instead of looping over models, and it never forms a V x V matrix for a
uniform LM: the boundary score lm(w | v) does not depend on v there, so the best
predecessor of every word is the best exit. Emissions are computed frame by
frame and only for states that can be alive at that frame (successors of live
states and word entries); any other state holds a LOG_ZERO-sized score that a
finite emission cannot change, so the search is the same as with the full
(T, n_total) matrix, which is never materialised.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .gmm import LOG_2PI
from .hmm import HMM, LOG_ZERO
from .lexicon import EOS, Lexicon

DEAD = LOG_ZERO / 2  # scores at or below this are unreachable hypotheses
EMISSION_CHUNK = 8192   # states per emission task; also bounds the temporaries
EMISSION_THREADS = min(4, os.cpu_count() or 1)


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

        # LM scores at word boundaries: rows are predecessors, columns
        # successors. Without a bigram log P(w | v) = log P(w) for every v, so
        # a single row stands for the whole matrix and none is built.
        self.uniform = not lexicon.bigram
        if self.uniform:
            self._lm = None
            self._lm_successor = np.array(
                [lexicon.log_prob(w, None) for w in self.names], dtype=np.float64
            )
        else:
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

        # Per-state and per-model arrays in the global index space.
        ms = [self.models[w] for w in self.names]
        self._starts = np.array([self.offsets[w] for w in self.names], dtype=np.int64)
        self._sizes = np.array([m.n_states for m in ms], dtype=np.int64)
        self._index = np.arange(total, dtype=np.int64)
        self._log_entry = np.concatenate([m.log_entry for m in ms])
        self._log_exit = np.concatenate([m.log_exit for m in ms])
        self._entry0 = np.array([m.log_entry[0] for m in ms], dtype=np.float64)

        # Arcs i -> i + d inside a model, one weight vector per offset d,
        # indexed by destination; LOG_ZERO where there is no such arc.
        arcs: dict[int, np.ndarray] = {}
        for m, off in zip(ms, self._starts):
            src, dst = np.nonzero(m.log_trans > DEAD)
            for d in np.unique(dst - src):
                sel = (dst - src) == d
                weights = arcs.setdefault(int(d), np.full(total, LOG_ZERO))
                weights[off + dst[sel]] = m.log_trans[src[sel], dst[sel]]
        # Largest offset first is smallest source first: the tie order of argmax.
        self._arcs = [(d, weights, weights > DEAD)
                      for d, weights in sorted(arcs.items(), reverse=True)]

        # Every state's GMM stacked by mixture size, with the same per-GMM
        # constants GMM.log_prob uses, so stacked emissions match it exactly.
        groups: dict[tuple[int, int], list] = {}
        for m, off in zip(ms, self._starts):
            for i, g in enumerate(m.states):
                groups.setdefault((g.n_components, g.n_features), []).append((off + i, g))
        self._group_of = np.zeros(total, dtype=np.int64)
        self._row_of = np.zeros(total, dtype=np.int64)
        self._gmms = []
        for gi, ((_, n_features), items) in enumerate(groups.items()):
            idx = np.array([j for j, _ in items], dtype=np.int64)
            self._group_of[idx] = gi
            self._row_of[idx] = np.arange(len(items))
            with np.errstate(divide="ignore"):
                log_w = np.stack([np.log(np.maximum(g.weights, 1e-300)) for _, g in items])
            self._gmms.append((
                n_features * LOG_2PI,
                np.stack([g.means for _, g in items]),
                np.stack([g.variances for _, g in items]),
                np.stack([np.sum(np.log(g.variances), axis=1) for _, g in items]),
                log_w,
            ))
        self._pool: ThreadPoolExecutor | None = None

    # -- emissions -------------------------------------------------------

    def log_emissions(self, X: np.ndarray) -> np.ndarray:
        """Stack every model's frame likelihoods -> (T, n_total)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.empty((X.shape[0], self.n_total))
        for w in self.names:
            off = self.offsets[w]
            out[:, off : off + self.models[w].n_states] = self.models[w].log_emission(X)
        return out

    def _emissions_at(self, x: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Log-likelihood of frame x in the given states; 0.0 everywhere else.

        Large sets are split over a few threads (numpy releases the GIL); each
        state is still computed on its own, so the split changes no value.
        """
        out = np.zeros(self.n_total)
        groups = self._group_of[states]
        chunks = []
        for gi in range(len(self._gmms)):
            sel = states if len(self._gmms) == 1 else states[groups == gi]
            chunks += [(gi, sel[k : k + EMISSION_CHUNK])
                       for k in range(0, sel.size, EMISSION_CHUNK)]
        if len(chunks) > 1:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(EMISSION_THREADS)
            list(self._pool.map(lambda c: self._emit(x, c[0], c[1], out), chunks))
        elif chunks:
            self._emit(x, chunks[0][0], chunks[0][1], out)
        return out

    def _emit(self, x: np.ndarray, gi: int, sel: np.ndarray, out: np.ndarray) -> None:
        """GMM.log_prob of frame x in states `sel`: same operations, same order."""
        const, means, variances, log_det, log_w = self._gmms[gi]
        rows = self._row_of[sel]
        diff = means[rows]                       # a copy, then (x - mu)^2 / var in place
        np.subtract(x[None, None, :], diff, out=diff)
        np.multiply(diff, diff, out=diff)
        np.divide(diff, variances[rows], out=diff)
        maha = np.sum(diff, axis=2)
        lp = -0.5 * (const + log_det[rows] + maha)
        out[sel] = logsumexp(lp + log_w[rows], axis=1)

    # -- search ----------------------------------------------------------

    def decode(self, X: np.ndarray) -> DecodeResult:
        """Return the most likely word sequence for feature matrix X."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        T = X.shape[0]
        if T == 0:
            return DecodeResult()

        cfg = self.config
        n, starts = self.n_total, self._starts
        lm = cfg.lm_scale * (self._lm_successor if self.uniform else self._lm)

        # Word link records: (word index, frame it ended on, parent record).
        # The decoded transcript is read out by walking this chain backwards.
        links: list[tuple[int, int, int]] = []

        # t = 0: enter every model from the sentence-start context.
        entry = (self._log_entry + np.repeat(cfg.lm_scale * self._lm_init, self._sizes)
                 + cfg.word_penalty)
        delta = entry + self._emissions_at(X[0], np.flatnonzero(self._log_entry > DEAD))
        bp = np.full(n, -1, dtype=np.int64)   # -> word link record

        for t in range(1, T):
            # 1. Score exiting each model at t-1.
            exit_score, exit_state = self._segment_max(delta + self._log_exit)
            exit_bp = bp[exit_state]

            # 2. Best predecessor of every word, which is exact for a bigram
            #    because the LM score is applied at the boundary and the
            #    predecessor identity is recorded.
            if self.uniform:
                v_best = int(exit_score.argmax())
                best_score = exit_score[v_best] + lm + cfg.word_penalty
            else:
                transition = exit_score[:, None] + lm + cfg.word_penalty
                best_prev = transition.argmax(axis=0)            # (V,) over v
                best_score = transition.max(axis=0)
            entering = np.flatnonzero(best_score > DEAD)

            # 3. Emissions only where a state can be alive: a successor of a
            #    live state, or the entry of a word that may be entered.
            live = delta > DEAD
            needed = np.zeros(n, dtype=bool)
            for d, _, arc in self._arcs:
                needed |= _shift(live, d, False) & arc
            needed[starts[entering]] = True
            logb = self._emissions_at(X[t], np.flatnonzero(needed))

            # 4. Propagate inside each model (ordinary Viterbi recursion).
            best = np.full(n, LOG_ZERO)
            source = self._index
            for d, weights, _ in self._arcs:
                cand = _shift(delta, d, LOG_ZERO) + weights
                better = cand > best
                best = np.where(better, cand, best)
                source = np.where(better, self._index - d, source)
            new_delta = best + logb
            new_bp = bp[source]

            # 5. Cross word boundaries.
            heads = starts[entering]
            start = best_score[entering] + self._entry0[entering] + logb[heads]
            wins = start > new_delta[heads]   # otherwise staying inside scores better
            entering, heads, start = entering[wins], heads[wins], start[wins]
            if entering.size:
                if self.uniform:
                    prev = _uniform_predecessors(exit_score, v_best, lm[entering],
                                                 cfg.word_penalty)
                else:
                    prev = best_prev[entering]
                # One link record per predecessor, numbered in order of first use.
                used, first, which = np.unique(prev, return_index=True, return_inverse=True)
                order = np.argsort(first, kind="stable")
                rank = np.empty_like(order)
                rank[order] = np.arange(order.size)
                base = len(links)
                links.extend((int(v), t - 1, int(exit_bp[v])) for v in used[order])
                new_delta[heads] = start
                new_bp[heads] = base + rank[which.ravel()]

            delta, bp = new_delta, new_bp
            delta = self._prune(delta)

        return self._traceback(delta, bp, links, T)

    def _segment_max(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per model: max of its slice of `values` and the index of its first maximum."""
        best = np.maximum.reduceat(values, self._starts)
        hit = values == np.repeat(best, self._sizes)
        first = np.minimum.reduceat(np.where(hit, self._index, self.n_total), self._starts)
        return best, first

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
        final = np.repeat(self.config.lm_scale * self._lm_final, self._sizes)
        final_score, final_state = self._segment_max(delta + self._log_exit + final)
        final_bp = bp[final_state]

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


def _shift(values: np.ndarray, d: int, fill) -> np.ndarray:
    """out[j] = values[j - d], `fill` where j - d falls outside the array."""
    if d == 0:
        return values
    out = np.full_like(values, fill)
    if d > 0:
        out[d:] = values[:-d]
    else:
        out[:d] = values[-d:]
    return out


def _uniform_predecessors(exit_score: np.ndarray, v_best: int, lm: np.ndarray,
                          word_penalty: float) -> np.ndarray:
    """Best predecessor of each entered word when lm(w | v) = lm[w] for every v.

    That is the best exit, except that the dense argmax keeps the first of tied
    maxima and rounding exit + lm[w] can tie an exit marginally below the best;
    such near-best exits are rechecked with the dense expression.
    """
    top = exit_score[v_best]
    tol = 1e-9 * (abs(top) + float(np.abs(lm).max()) + abs(word_penalty) + 1.0)
    near = np.flatnonzero(exit_score >= top - tol)
    if near.size == 1:
        return np.full(lm.size, v_best, dtype=np.int64)
    totals = exit_score[near][:, None] + lm[None, :] + word_penalty
    return near[totals.argmax(axis=0)]


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
