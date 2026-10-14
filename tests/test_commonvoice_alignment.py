import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import script
from script import match_word_indices, save_wav, valid_word_bounds


def _script_function(name):
    function = getattr(script, name, None)
    assert callable(function), f"script.{name} must be implemented"
    return function


def test_matches_complete_sentence():
    assert match_word_indices(["the", "quick"], ["the", "quick"]) == [
        (0, 0),
        (1, 1),
    ]


def test_skips_reference_word_missing_from_whisper():
    assert match_word_indices(
        ["the", "quick", "brown", "fox"], ["the", "quick", "fox"]
    ) == [(0, 0), (1, 1), (3, 2)]


def test_ignores_word_added_by_whisper():
    assert match_word_indices(
        ["the", "quick", "fox"], ["the", "very", "quick", "fox"]
    ) == [(0, 0), (1, 2), (2, 3)]


def test_skips_substituted_word():
    assert match_word_indices(["a", "cat"], ["a", "dog"]) == [(0, 0)]


def test_matches_repeated_words_without_reusing_indices():
    assert match_word_indices(["go", "go", "home"], ["go", "go", "home"]) == [
        (0, 0),
        (1, 1),
        (2, 2),
    ]


def test_matches_after_case_and_punctuation_normalization():
    assert match_word_indices(["Hello,", "WORLD!"], ["hello", "world."]) == [
        (0, 0),
        (1, 1),
    ]


@pytest.mark.parametrize(
    "reference, whisper, expected",
    [
        ("We’re", "we're", [(0, 0)]),
        ("we're", "were", []),
        ("he’ll", "he'll", [(0, 0)]),
        ("he'll", "hell", []),
        ("Hello,", "hello", [(0, 0)]),
        ("hello-world", "helloworld", []),
        ("'We‘re!'", "we're", [(0, 0)]),
    ],
)
def test_comparison_normalization_preserves_internal_apostrophes(
    reference, whisper, expected
):
    assert match_word_indices([reference], [whisper]) == expected


def test_returns_no_matches_for_unrelated_sequences():
    assert match_word_indices(["alpha", "beta"], ["gamma", "delta"]) == []


def test_match_indices_are_monotone_and_unique():
    matches = match_word_indices(
        ["go", "slowly", "go", "home"], ["go", "quickly", "go", "home"]
    )
    reference_indices, whisper_indices = zip(*matches)

    assert list(reference_indices) == [0, 2, 3]
    assert list(whisper_indices) == [0, 2, 3]
    assert len(set(reference_indices)) == len(reference_indices)
    assert len(set(whisper_indices)) == len(whisper_indices)


def test_save_wav_does_not_overwrite_existing_clip(tmp_path):
    path = tmp_path / "word_0000.wav"
    path.write_bytes(b"existing clip")

    with pytest.raises(FileExistsError):
        save_wav(path, np.zeros(1600), 16000)

    assert path.read_bytes() == b"existing clip"


def test_main_rejects_word_file_before_wav_retry(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "commonvoice"
    clips_dir = data_dir / "validated"
    clips_dir.mkdir(parents=True)
    (data_dir / "validated.tsv").write_text("synthetic manifest", encoding="utf-8")
    (clips_dir / "sample.wav").write_bytes(b"synthetic audio")

    output_dir = tmp_path / "words"
    output_dir.mkdir()
    word_path = output_dir / "about"
    word_path.write_bytes(b"not a directory")

    class FakeManifest:
        def iterrows(self):
            return iter([(0, {"path": "sample.wav", "sentence": "about"})])

    class FakeWhisperModel:
        def transcribe(self, audio, language, word_timestamps, suppress_silence):
            assert suppress_silence is False
            result = SimpleNamespace(
                all_words=lambda: [SimpleNamespace(word="about", start=0.1, end=0.2)]
            )
            return result

    librosa = ModuleType("librosa")
    librosa.load = lambda path, sr, mono: (np.ones(sr // 2), sr)
    pandas = ModuleType("pandas")
    pandas.read_csv = lambda path, sep: FakeManifest()
    stable_whisper = ModuleType("stable_whisper")
    stable_whisper.load_model = lambda name, device: FakeWhisperModel()
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    for name, module in (
        ("librosa", librosa),
        ("pandas", pandas),
        ("stable_whisper", stable_whisper),
        ("torch", torch),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(script, "DATA_DIR", data_dir)
    monkeypatch.setattr(script, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(script, "TARGET_VOCABULARY", {"about"})
    monkeypatch.setattr(script, "CLIPS_PER_WORD", 1)
    save_attempts = []

    def record_save_attempt(path, audio, sr):
        save_attempts.append(path)
        if len(save_attempts) == 1:
            raise FileExistsError(path)
        raise OSError("stop a repeated retry")

    monkeypatch.setattr(script, "save_wav", record_save_attempt)

    script.main()

    assert save_attempts == []
    assert word_path.read_bytes() == b"not a directory"
    output = capsys.readouterr().out
    assert "cartella di output non valida" in output
    assert "Errori di salvataggio clip: 1" in output


def test_save_wav_removes_partial_file_and_preserves_existing_file(
    tmp_path, monkeypatch
):
    partial_path = tmp_path / "partial.wav"
    existing_path = tmp_path / "existing.wav"
    existing_path.write_bytes(b"existing clip")

    class FailingWaveWriter:
        def __init__(self, output):
            self.output = output

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def setnchannels(self, channels):
            pass

        def setsampwidth(self, width):
            pass

        def setframerate(self, rate):
            pass

        def close(self):
            pass

        def writeframes(self, frames):
            self.output.write(b"partial wav data")
            raise OSError("simulated write failure")

    monkeypatch.setattr(
        script.wave, "open", lambda output, mode: FailingWaveWriter(output)
    )

    with pytest.raises(OSError, match="simulated write failure"):
        save_wav(partial_path, np.zeros(1600), 16000)

    assert not partial_path.exists()
    with pytest.raises(FileExistsError):
        save_wav(existing_path, np.zeros(1600), 16000)
    assert existing_path.read_bytes() == b"existing clip"


def test_alignment_statistics_without_target_vocabulary():
    statistics = _script_function("alignment_statistics")(
        ["hello", "about"], ["hello"], [(0, 0)], None
    )

    assert statistics == {
        "commonvoice_words_considered": 2,
        "exact_matches": 1,
        "reference_words_discarded": 1,
        "whisper_words_ignored": 0,
    }


def test_alignment_statistics_consider_only_target_words_in_sentence():
    statistics = _script_function("alignment_statistics")(
        ["hello", "about", "world"], [], [], {"about"}
    )

    assert statistics == {
        "commonvoice_words_considered": 1,
        "exact_matches": 0,
        "reference_words_discarded": 1,
        "whisper_words_ignored": 0,
    }


def test_alignment_statistics_ignore_match_on_non_target_word():
    statistics = _script_function("alignment_statistics")(
        ["hello", "about"], ["hello"], [(0, 0)], {"about"}
    )

    assert statistics == {
        "commonvoice_words_considered": 1,
        "exact_matches": 0,
        "reference_words_discarded": 1,
        "whisper_words_ignored": 1,
    }


def test_alignment_statistics_count_target_word_match():
    statistics = _script_function("alignment_statistics")(
        ["hello", "about"], ["about"], [(1, 0)], {"about"}
    )

    assert statistics == {
        "commonvoice_words_considered": 1,
        "exact_matches": 1,
        "reference_words_discarded": 0,
        "whisper_words_ignored": 0,
    }


def test_alignment_statistics_count_repeated_target_occurrences():
    statistics = _script_function("alignment_statistics")(
        ["about", "hello", "about"],
        ["about", "about"],
        [(0, 0), (2, 1)],
        {"about"},
    )

    assert statistics == {
        "commonvoice_words_considered": 2,
        "exact_matches": 2,
        "reference_words_discarded": 0,
        "whisper_words_ignored": 0,
    }


def test_accepts_valid_word_bounds_at_audio_edges():
    assert valid_word_bounds(0.0, 2.0, 2.0) == (0.0, 2.0)


@pytest.mark.parametrize(
    "start, end, duration",
    [
        (None, 1.0, 2.0),
        (0.0, None, 2.0),
        ("0.0", 1.0, 2.0),
        (float("nan"), 1.0, 2.0),
        (0.0, float("inf"), 2.0),
        (1.0, 1.0, 2.0),
        (1.1, 1.0, 2.0),
        (-0.1, 0.5, 2.0),
        (0.0, 2.1, 2.0),
    ],
)
def test_rejects_invalid_word_bounds(start, end, duration):
    assert valid_word_bounds(start, end, duration) is None
