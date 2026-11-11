"""Model persistence without pickle.

Loading a model must never execute code from the file, so nothing here uses
pickle. Nested structure goes to JSON, numpy arrays go to an npz side-car, and
the two are stitched back together by path.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np

ARRAY_TAG = "__ndarray__:"


def _flatten(obj, prefix, arrays):
    """Replace every ndarray with a tagged key and collect it in `arrays`."""
    if isinstance(obj, np.ndarray):
        key = prefix or "root"
        arrays[key] = obj
        return ARRAY_TAG + key
    if isinstance(obj, dict):
        return {k: _flatten(v, f"{prefix}/{k}" if prefix else str(k), arrays)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_flatten(v, f"{prefix}/{i}", arrays) for i, v in enumerate(obj)]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _unflatten(obj, arrays):
    if isinstance(obj, str) and obj.startswith(ARRAY_TAG):
        return arrays[obj[len(ARRAY_TAG):]]
    if isinstance(obj, dict):
        return {k: _unflatten(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unflatten(v, arrays) for v in obj]
    return obj


def save(path, payload: dict) -> Path:
    """Write `payload` (nested dicts/lists/arrays) to a single .psm file."""
    path = Path(path)
    arrays: dict[str, np.ndarray] = {}
    skeleton = _flatten(payload, "", arrays)

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("structure.json", json.dumps(skeleton, indent=1))
        z.writestr("arrays.npz", buf.getvalue())
    return path


def load(path) -> dict:
    """Read a .psm file back into the original nested structure."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path, "r") as z:
        skeleton = json.loads(z.read("structure.json"))
        with np.load(io.BytesIO(z.read("arrays.npz")), allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
    return _unflatten(skeleton, arrays)
