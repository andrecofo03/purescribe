"""Regression test for the phase-2 per-word clip quota defaults."""

import commonvoice_purescribe as cv


def test_extract_defaults_keep_12_to_23_clips_per_word(monkeypatch):
    seen = {}
    monkeypatch.setattr(cv, "load_settings", lambda path: {})
    monkeypatch.setattr(cv, "apply_settings", lambda args, settings: None)
    monkeypatch.setattr(cv, "extract", lambda args: seen.update(vars(args)))

    assert cv.main(["extract"]) == 0
    assert seen["clips_per_word"] == 23
    assert seen["min_clips_per_word"] == 12
