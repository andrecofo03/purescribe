"""PureScribe - speech recognition from first principles.

No neural networks. No language models. No network calls. The entire signal
chain is explicit numpy: pre-emphasis, framing, FFT, mel filterbank, cepstral
compression, then Gaussian-mixture HMMs decoded with Viterbi.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Submodules are resolved on first attribute access (PEP 562). Keeps
# `import purescribe` cheap and lets modules import each other freely.
_EXPORTS = {
    "load_audio": "purescribe.audio",
    "Waveform": "purescribe.audio",
    "FeatureConfig": "purescribe.features",
    "mfcc": "purescribe.features",
    "GMM": "purescribe.gmm",
    "HMM": "purescribe.hmm",
    "Lexicon": "purescribe.lexicon",
    "Recognizer": "purescribe.recognizer",
    "Transcript": "purescribe.recognizer",
}

__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'purescribe' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__():
    return list(__all__)


if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .audio import Waveform, load_audio
    from .features import FeatureConfig, mfcc
    from .gmm import GMM
    from .hmm import HMM
    from .lexicon import Lexicon
    from .recognizer import Recognizer, Transcript
