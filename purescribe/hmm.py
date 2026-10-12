"""Left-to-right hidden Markov models with Gaussian-mixture emissions.

One model per word. States are traversed in order with self-loops, which is
what encodes that an utterance has a direction: you may dwell in a state while
a sound is sustained, or advance to the next, but never go back. The self-loop
probability implicitly models duration as a geometric distribution.

Notation throughout: T frames, N emitting states, logB is (T, N) emission
log-likelihoods. Everything is in the log domain.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .gmm import GMM, VAR_FLOOR

LOG_ZERO = -1e30  # stands in for log(0); finite so arithmetic stays well-defined


class HMM:
    """A left-to-right HMM over `n_states` emitting states."""

    def __init__(
        self,
        n_states: int,
        n_features: int,
        n_mixtures: int = 1,
        name: str = "",
        allow_skip: bool = False,
        var_floor: float = VAR_FLOOR,
    ):
        self.n_states = int(n_states)
        self.n_features = int(n_features)
        self.n_mixtures = int(n_mixtures)
        self.name = name
        self.allow_skip = bool(allow_skip)
        self.var_floor = float(var_floor)

        self.states = [GMM(n_mixtures, n_features, var_floor)
                       for _ in range(self.n_states)]
        self.log_trans = self._init_transitions()

        # Non-emitting entry/exit, HTK style. Left-to-right means entry is
        # always state 0; exit is only from the final state unless skips are on.
        self.log_entry = np.full(self.n_states, LOG_ZERO)
        self.log_entry[0] = 0.0
        self.log_exit = np.full(self.n_states, LOG_ZERO)
        self.log_exit[-1] = np.log(0.5)

    def _init_transitions(self) -> np.ndarray:
        """Self-loop and advance at 0.5 each; optionally a skip arc."""
        A = np.full((self.n_states, self.n_states), LOG_ZERO)
        for i in range(self.n_states):
            targets = [i]
            if i + 1 < self.n_states:
                targets.append(i + 1)
            if self.allow_skip and i + 2 < self.n_states:
                targets.append(i + 2)
            p = np.log(1.0 / len(targets))
            for j in targets:
                A[i, j] = p
        return A

    @property
    def min_duration(self) -> int:
        """Fewest frames the model can emit - one per state without skips."""
        return 1 if self.allow_skip else self.n_states

    # -- emissions -------------------------------------------------------

    def log_emission(self, X: np.ndarray) -> np.ndarray:
        """Emission log-likelihood of every frame in every state -> (T, N)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.empty((X.shape[0], self.n_states))
        for i, gmm in enumerate(self.states):
            out[:, i] = gmm.log_prob(X)
        return out

    # -- inference -------------------------------------------------------

    def forward(self, logB: np.ndarray):
        """Scaled-free log-domain forward pass -> (alpha, log-likelihood)."""
        T = logB.shape[0]
        alpha = np.full((T, self.n_states), LOG_ZERO)
        alpha[0] = self.log_entry + logB[0]
        for t in range(1, T):
            # logsumexp over the source state i for every destination j.
            alpha[t] = logsumexp(alpha[t - 1][:, None] + self.log_trans, axis=0) + logB[t]
        ll = float(logsumexp(alpha[T - 1] + self.log_exit))
        return alpha, ll

    def backward(self, logB: np.ndarray) -> np.ndarray:
        """Log-domain backward pass -> beta of shape (T, N)."""
        T = logB.shape[0]
        beta = np.full((T, self.n_states), LOG_ZERO)
        beta[T - 1] = self.log_exit
        for t in range(T - 2, -1, -1):
            beta[t] = logsumexp(self.log_trans + (logB[t + 1] + beta[t + 1])[None, :], axis=1)
        return beta

    def viterbi(self, logB: np.ndarray):
        """Most likely state sequence -> (path, score).

        Same recursion as forward with max in place of sum. The backpointer
        array is what turns a score into an alignment, which is how training
        gets its frame-to-state assignment.
        """
        T = logB.shape[0]
        delta = np.full((T, self.n_states), LOG_ZERO)
        psi = np.zeros((T, self.n_states), dtype=np.int32)

        delta[0] = self.log_entry + logB[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + self.log_trans   # (from, to)
            psi[t] = scores.argmax(axis=0)
            delta[t] = scores.max(axis=0) + logB[t]

        final = delta[T - 1] + self.log_exit
        path = np.zeros(T, dtype=np.int32)
        path[T - 1] = int(final.argmax())
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, float(final.max())

    def score(self, X: np.ndarray, use_viterbi: bool = False) -> float:
        """Log-likelihood that this model produced X."""
        X = np.atleast_2d(X)
        if X.shape[0] == 0:
            return LOG_ZERO
        logB = self.log_emission(X)
        if use_viterbi:
            return self.viterbi(logB)[1]
        return self.forward(logB)[1]

    def align(self, X: np.ndarray) -> np.ndarray:
        """Frame -> state assignment under the best path."""
        return self.viterbi(self.log_emission(np.atleast_2d(X)))[0]

    # -- training --------------------------------------------------------

    def init_uniform(self, sequences: list[np.ndarray], seed: int = 0) -> "HMM":
        """Flat start: chop every utterance into N equal parts, one per state.

        No alignment exists before training, so we assume each state occupies an
        equal share of the utterance. Crude, but it puts every state in roughly
        the right region of feature space for Viterbi training to refine.
        """
        pooled = [[] for _ in range(self.n_states)]
        for X in sequences:
            X = np.atleast_2d(X)
            T = X.shape[0]
            if T == 0:
                continue
            # Boundaries of N contiguous, near-equal segments.
            edges = np.linspace(0, T, self.n_states + 1).astype(int)
            for i in range(self.n_states):
                lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
                pooled[i].append(X[lo:min(hi, T)])

        allX = np.vstack([np.atleast_2d(x) for x in sequences if np.atleast_2d(x).size])
        for i, chunks in enumerate(pooled):
            data = np.vstack(chunks) if chunks else allX
            # Force: this is the global statistic every state's floor derives
            # from, and it must survive later re-fits on state-local data.
            self.states[i].set_floor_from(allX, force=True)
            self.states[i].fit(data, n_iter=10, seed=seed + i)
        return self

    def fit(
        self,
        sequences: list[np.ndarray],
        n_iter: int = 8,
        method: str = "viterbi",
        seed: int = 0,
        init: bool = True,
        tol: float = 1e-4,
    ) -> "HMM":
        """Re-estimate from a list of utterances, each (T_i, n_features).

        `viterbi` is segmental k-means: hard-assign each frame to its best state
        and refit. Fast and very stable on small corpora. `baum_welch` uses soft
        occupancies, squeezing more out of limited data at some risk of drifting
        into a poor local optimum.
        """
        sequences = [np.atleast_2d(np.asarray(s, dtype=np.float64)) for s in sequences]
        sequences = [s for s in sequences if s.shape[0] >= 1]
        if not sequences:
            raise ValueError(f"no training data for model {self.name!r}")

        if init:
            self.init_uniform(sequences, seed=seed)

        prev = -np.inf
        for _ in range(n_iter):
            if method == "viterbi":
                ll = self._viterbi_step(sequences, seed)
            elif method == "baum_welch":
                ll = self._baum_welch_step(sequences)
            else:
                raise ValueError(f"unknown training method: {method}")
            if np.isfinite(prev) and abs(ll - prev) < tol * max(1.0, abs(prev)):
                break
            prev = ll
        return self

    def _viterbi_step(self, sequences: list[np.ndarray], seed: int) -> float:
        """One segmental k-means iteration: align, regroup, refit."""
        assigned = [[] for _ in range(self.n_states)]
        trans_counts = np.zeros((self.n_states, self.n_states))
        exit_counts = np.zeros(self.n_states)
        total_ll = 0.0

        for X in sequences:
            logB = self.log_emission(X)
            path, ll = self.viterbi(logB)
            total_ll += ll
            for t, s in enumerate(path):
                assigned[s].append(X[t])
            for t in range(1, len(path)):
                trans_counts[path[t - 1], path[t]] += 1
            exit_counts[path[-1]] += 1

        for i, frames in enumerate(assigned):
            if frames:
                self.states[i].fit(np.vstack(frames), n_iter=10, seed=seed + i)
            # A state that won no frames keeps its parameters; dropping it
            # would change the topology mid-training.

        self._update_transitions(trans_counts, exit_counts)
        return total_ll / max(len(sequences), 1)

    def _baum_welch_step(self, sequences: list[np.ndarray]) -> float:
        """One EM iteration with soft state occupancies."""
        K, D = self.n_mixtures, self.n_features
        nk = np.zeros((self.n_states, K))
        sx = np.zeros((self.n_states, K, D))
        sxx = np.zeros((self.n_states, K, D))
        trans_counts = np.zeros((self.n_states, self.n_states))
        exit_counts = np.zeros(self.n_states)
        total_ll = 0.0

        for X in sequences:
            T = X.shape[0]
            logB = self.log_emission(X)
            alpha, ll = self.forward(logB)
            beta = self.backward(logB)
            total_ll += ll

            gamma = np.exp(alpha + beta - ll)            # (T, N) state occupancy

            for i in range(self.n_states):
                w = gamma[:, i]
                if w.sum() <= 1e-8:
                    continue
                # Split state occupancy across the mixture components.
                a, b, c = self.states[i].accumulate(X, frame_weights=w)
                nk[i] += a
                sx[i] += b
                sxx[i] += c

            if T > 1:
                # xi[t,i,j]: probability of the i->j arc between t and t+1.
                for t in range(T - 1):
                    xi = (alpha[t][:, None] + self.log_trans
                          + (logB[t + 1] + beta[t + 1])[None, :] - ll)
                    trans_counts += np.exp(xi)
            exit_counts += np.exp(alpha[T - 1] + self.log_exit - ll)

        for i in range(self.n_states):
            if nk[i].sum() > 1e-8:
                self.states[i].set_from_stats(nk[i], sx[i], sxx[i])

        self._update_transitions(trans_counts, exit_counts)
        return total_ll / max(len(sequences), 1)

    def _update_transitions(self, trans_counts: np.ndarray, exit_counts: np.ndarray) -> None:
        """Re-normalise arc counts into log probabilities, topology preserved."""
        allowed = self.log_trans > LOG_ZERO / 2
        trans_counts = np.where(allowed, trans_counts, 0.0)

        for i in range(self.n_states):
            total = trans_counts[i].sum() + exit_counts[i]
            if total <= 0:
                continue  # no evidence: keep the prior
            row = np.full(self.n_states, LOG_ZERO)
            nz = allowed[i] & (trans_counts[i] > 0)
            row[nz] = np.log(trans_counts[i][nz] / total)
            self.log_trans[i] = row
            if exit_counts[i] > 0 and self.log_exit[i] > LOG_ZERO / 2:
                self.log_exit[i] = float(np.log(exit_counts[i] / total))

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "n_states": self.n_states,
            "n_features": self.n_features,
            "n_mixtures": self.n_mixtures,
            "name": self.name,
            "allow_skip": self.allow_skip,
            "var_floor": self.var_floor,
            "log_trans": self.log_trans,
            "log_entry": self.log_entry,
            "log_exit": self.log_exit,
            "states": [g.to_dict() for g in self.states],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HMM":
        h = cls(int(d["n_states"]), int(d["n_features"]), int(d["n_mixtures"]),
                str(d["name"]), bool(d["allow_skip"]),
                float(d.get("var_floor", VAR_FLOOR)))
        h.log_trans = np.asarray(d["log_trans"], dtype=np.float64)
        h.log_entry = np.asarray(d["log_entry"], dtype=np.float64)
        h.log_exit = np.asarray(d["log_exit"], dtype=np.float64)
        h.states = [GMM.from_dict(s) for s in d["states"]]
        return h

    def __repr__(self) -> str:
        return (f"HMM(name={self.name!r}, states={self.n_states}, "
                f"mix={self.n_mixtures}, dim={self.n_features})")
