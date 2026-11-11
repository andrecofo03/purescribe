"""Vocabulary and the statistical grammar over it.

This is a *language model* in the original, pre-neural sense: a bigram over a
closed vocabulary, estimated by counting. No neural network, no generation -
just P(word | previous word) used to bias the search. On a constrained task it
removes most of the errors that acoustics alone cannot resolve.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

SIL = "<sil>"    # silence / background model, decoded but never printed
BOS = "<s>"      # sentence start context
EOS = "</s>"     # sentence end context

LOG_ZERO = -1e30


class Lexicon:
    """Closed vocabulary plus a smoothed bigram over it."""

    def __init__(self, words, silence: str = SIL):
        self.silence = silence
        words = [w for w in dict.fromkeys(words) if w != silence]
        self.words = list(words)
        self.all_models = self.words + [silence]

        self.unigram: dict[str, float] = {}
        self.bigram: dict[tuple[str, str], float] = {}
        self._uniform()

    def _uniform(self) -> None:
        """Flat prior: every word equally likely, no grammar."""
        # Silence counts as one more loop member, hence the +1.
        p = math.log(1.0 / max(len(self.words) + 1, 1))
        self.unigram = {w: p for w in self.words}
        self.silence_log_prob = p
        self.bigram = {}

    # -- estimation ------------------------------------------------------

    def fit(self, sentences, discount: float = 0.5) -> "Lexicon":
        """Count bigrams over training transcripts with absolute discounting.

        Absolute discounting subtracts a fixed mass from every seen bigram and
        redistributes it over the unigram, so an unseen word pair gets a small
        but non-zero probability instead of blocking the path outright.
        """
        uni = Counter()
        bi = Counter()
        context = Counter()

        for sent in sentences:
            tokens = [w for w in sent if w != self.silence]
            if not tokens:
                continue
            uni.update(tokens)
            padded = [BOS] + tokens + [EOS]
            for a, b in zip(padded, padded[1:]):
                bi[(a, b)] += 1
                context[a] += 1

        total = sum(uni.values())
        if total == 0:
            self._uniform()
            return self

        # Add-one unigram, so a vocabulary word never seen in the transcripts
        # still has mass.
        v = len(self.words)
        self.unigram = {
            w: math.log((uni.get(w, 0) + 1.0) / (total + v)) for w in self.words
        }
        # Silence is as likely as an average word; it must not be free.
        self.silence_log_prob = float(np.mean(list(self.unigram.values()))) \
            if self.unigram else math.log(0.5)

        self.bigram = {}
        for (a, b), c in bi.items():
            if b not in self.unigram and b != EOS:
                continue
            denom = context[a]
            if denom <= 0:
                continue
            p = max(c - discount, 1e-6) / denom
            self.bigram[(a, b)] = math.log(p)

        # Mass reserved per context for the unigram backoff.
        self._backoff = {}
        for a, n in context.items():
            types = sum(1 for (x, _) in bi if x == a)
            self._backoff[a] = math.log(max(discount * types / n, 1e-6))
        return self

    # -- scoring ---------------------------------------------------------

    def log_prob(self, word: str, prev: str | None = None) -> float:
        """log P(word | prev).

        Silence is scored as an ordinary member of the loop. Making it free was
        a bug: with the LM scale applied, a real word costs lm_scale*log(1/V)
        while free silence costs nothing, so the decoder deletes every word and
        explains the whole recording as background. Silence must compete on the
        same terms as any other model.
        """
        # if word == self.silence:
        #     return 0.0
        if word == self.silence:
            return self.silence_log_prob
        if prev is None or prev == self.silence:
            prev = BOS
        if not self.bigram:
            return self.unigram.get(word, LOG_ZERO)
        key = (prev, word)
        if key in self.bigram:
            return self.bigram[key]
        backoff = getattr(self, "_backoff", {}).get(prev, math.log(0.4))
        return backoff + self.unigram.get(word, math.log(1e-6))

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, w) -> bool:
        return w in self.words

    def to_dict(self) -> dict:
        return {
            "words": self.words,
            "silence": self.silence,
            "unigram": self.unigram,
            "bigram": {f"{a}\t{b}": v for (a, b), v in self.bigram.items()},
            "backoff": getattr(self, "_backoff", {}),
            "silence_log_prob": self.silence_log_prob,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Lexicon":
        lex = cls(list(d["words"]), str(d["silence"]))
        lex.unigram = dict(d["unigram"])
        lex.bigram = {tuple(k.split("\t")): v for k, v in dict(d["bigram"]).items()}
        lex._backoff = dict(d.get("backoff", {}))
        lex.silence_log_prob = float(d.get("silence_log_prob", math.log(0.5)))
        return lex
