"""Diagonal-covariance Gaussian mixtures - the emission density of each HMM state.

Why diagonal: a full covariance in 39 dimensions needs 780 free parameters per
component, which no small corpus can estimate. The DCT in the front end already
decorrelated the features, so off-diagonal terms are small and a diagonal model
buys robustness at little cost in likelihood.

Everything is computed in the log domain. Frame likelihoods underflow float64
within a few dozen frames otherwise.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

LOG_2PI = float(np.log(2.0 * np.pi))
VAR_FLOOR = 1e-3  # relative to the global feature variance


class GMM:
    """A mixture of `n_components` diagonal Gaussians over `n_features` dims."""

    def __init__(self, n_components: int, n_features: int, var_floor: float = VAR_FLOOR):
        self.n_components = int(n_components)
        self.n_features = int(n_features)
        self.var_floor = float(var_floor)

        self.weights = np.full(self.n_components, 1.0 / self.n_components)
        self.means = np.zeros((self.n_components, self.n_features))
        self.variances = np.ones((self.n_components, self.n_features))
        self._floor = np.full(self.n_features, 1e-6)
        self._floor_set = False

    # -- likelihood ------------------------------------------------------

    def log_prob_components(self, X: np.ndarray) -> np.ndarray:
        """Per-component log density -> (n_samples, n_components).

        log N(x | mu, diag(v)) = -0.5 * [ D log 2pi + sum log v + sum (x-mu)^2 / v ]
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        var = self.variances
        # (n, 1, D) - (1, K, D) -> (n, K, D), then reduce over D.
        diff = X[:, None, :] - self.means[None, :, :]
        maha = np.sum((diff * diff) / var[None, :, :], axis=2)
        log_det = np.sum(np.log(var), axis=1)
        return -0.5 * (self.n_features * LOG_2PI + log_det[None, :] + maha)

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        """Mixture log-likelihood of each sample -> (n_samples,)."""
        lp = self.log_prob_components(X)
        with np.errstate(divide="ignore"):
            log_w = np.log(np.maximum(self.weights, 1e-300))
        return logsumexp(lp + log_w[None, :], axis=1)

    def posteriors(self, X: np.ndarray) -> np.ndarray:
        """Responsibility of each component for each sample -> (n_samples, K)."""
        with np.errstate(divide="ignore"):
            log_w = np.log(np.maximum(self.weights, 1e-300))
        joint = self.log_prob_components(X) + log_w[None, :]
        return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))

    # -- estimation ------------------------------------------------------

    def set_floor_from(self, X: np.ndarray, force: bool = False) -> None:
        """Set the variance floor relative to the global spread of the data.

        A component that captures a handful of near-identical frames will drive
        its variance towards zero and its likelihood towards +inf, hijacking the
        model. Flooring is the standard defence.

        The floor must come from *global* statistics and then stay fixed. An
        earlier version re-derived it inside every fit() from whichever frames
        the state currently owned, so as a state's data grew more homogeneous
        its floor shrank with it - the floor collapsed along with the variance
        it was supposed to hold up. Once set, it is kept unless forced.
        """
        X = np.atleast_2d(X)
        if X.shape[0] < 2:
            return
        if self._floor_set and not force:
            return
        self._floor = np.maximum(X.var(axis=0) * self.var_floor, 1e-8)
        self._floor_set = True

    def init_from_data(self, X: np.ndarray, seed: int = 0) -> "GMM":
        """Initialise by k-means, then one moment-matching M-step."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self.set_floor_from(X)   # no-op once a global floor has been set

        if X.shape[0] <= self.n_components:
            # Degenerate but legal: seed every component from the global stats.
            self.means[:] = X.mean(axis=0) if X.size else 0.0
            self.variances[:] = np.maximum(
                X.var(axis=0) if X.shape[0] > 1 else np.ones(self.n_features), self._floor
            )
            self.weights[:] = 1.0 / self.n_components
            return self

        labels, centres = _kmeans(X, self.n_components, seed=seed)
        for k in range(self.n_components):
            mask = labels == k
            n = int(mask.sum())
            if n == 0:
                self.means[k] = X[np.random.default_rng(seed + k).integers(X.shape[0])]
                self.variances[k] = np.maximum(X.var(axis=0), self._floor)
                self.weights[k] = 1.0 / X.shape[0]
            else:
                self.means[k] = X[mask].mean(axis=0)
                self.variances[k] = np.maximum(X[mask].var(axis=0), self._floor)
                self.weights[k] = n / X.shape[0]
        self.weights /= self.weights.sum()
        return self

    def fit(self, X: np.ndarray, n_iter: int = 20, tol: float = 1e-4,
            seed: int = 0, init: bool = True) -> "GMM":
        """Expectation-maximisation. Returns self; monotone in log-likelihood."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if init:
            self.init_from_data(X, seed=seed)
        if X.shape[0] < 2:
            return self

        prev = -np.inf
        for _ in range(n_iter):
            resp = self.posteriors(X)                    # E-step
            self._m_step(X, resp)                        # M-step
            ll = float(self.log_prob(X).mean())
            if np.isfinite(prev) and abs(ll - prev) < tol * max(1.0, abs(prev)):
                break
            prev = ll
        return self

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        """Closed-form weighted moments. `resp` is (n_samples, n_components)."""
        nk = resp.sum(axis=0)                            # soft count per component
        safe = np.maximum(nk, 1e-10)

        means = (resp.T @ X) / safe[:, None]
        # E[x^2] - E[x]^2, computed with the same responsibilities.
        sq = (resp.T @ (X * X)) / safe[:, None]
        variances = np.maximum(sq - means * means, self._floor[None, :])

        # Components that collected no mass keep their previous parameters
        # rather than collapsing to NaN.
        alive = nk > 1e-8
        self.means[alive] = means[alive]
        self.variances[alive] = variances[alive]
        self.weights = np.maximum(nk, 1e-10) / max(nk.sum(), 1e-10)

    def accumulate(self, X: np.ndarray, frame_weights: np.ndarray | None = None):
        """Return sufficient statistics for a distributed / embedded M-step.

        Used by HMM training, where a frame belongs to a state only partially.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        resp = self.posteriors(X)
        if frame_weights is not None:
            resp = resp * np.asarray(frame_weights, dtype=np.float64)[:, None]
        nk = resp.sum(axis=0)
        return nk, resp.T @ X, resp.T @ (X * X)

    def set_from_stats(self, nk: np.ndarray, sx: np.ndarray, sxx: np.ndarray) -> None:
        """Apply an M-step from accumulated statistics."""
        safe = np.maximum(nk, 1e-10)
        means = sx / safe[:, None]
        variances = np.maximum(sxx / safe[:, None] - means * means, self._floor[None, :])
        alive = nk > 1e-8
        if not alive.any():
            return
        self.means[alive] = means[alive]
        self.variances[alive] = variances[alive]
        self.weights = np.maximum(nk, 1e-10) / max(nk.sum(), 1e-10)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "n_components": self.n_components,
            "n_features": self.n_features,
            "var_floor": self.var_floor,
            "weights": self.weights,
            "means": self.means,
            "variances": self.variances,
            "floor": self._floor,
            "floor_set": bool(self._floor_set),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GMM":
        g = cls(int(d["n_components"]), int(d["n_features"]), float(d["var_floor"]))
        g.weights = np.asarray(d["weights"], dtype=np.float64)
        g.means = np.asarray(d["means"], dtype=np.float64)
        g.variances = np.asarray(d["variances"], dtype=np.float64)
        g._floor = np.asarray(d["floor"], dtype=np.float64)
        g._floor_set = bool(d.get("floor_set", True))
        return g


def _kmeans(X: np.ndarray, k: int, n_iter: int = 30, seed: int = 0):
    """k-means++ initialisation followed by Lloyd iterations."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # k-means++: each new centre is drawn with probability proportional to its
    # squared distance from the nearest existing centre. Avoids the degenerate
    # starts that plain random seeding produces on clustered speech data.
    centres = np.empty((k, X.shape[1]))
    centres[0] = X[rng.integers(n)]
    closest = np.sum((X - centres[0]) ** 2, axis=1)
    for i in range(1, k):
        total = closest.sum()
        probs = closest / total if total > 0 else np.full(n, 1.0 / n)
        centres[i] = X[rng.choice(n, p=probs)]
        closest = np.minimum(closest, np.sum((X - centres[i]) ** 2, axis=1))

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d = np.sum((X[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centres[i] = X[mask].mean(axis=0)
    return labels, centres
