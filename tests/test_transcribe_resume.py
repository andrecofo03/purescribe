"""Phase 5 (lost_in_translation.py transcribe): progress output and --resume.

The recognizer and the audio loader are replaced by fakes, so no model, ffmpeg or
DeepSpeak data is needed.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lost_in_translation as lit  # noqa: E402
import purescribe.audio  # noqa: E402
import purescribe.recognizer  # noqa: E402
from purescribe.audio import Waveform  # noqa: E402

VIDEOS = ["real/a1.mp4", "real/a2.mp4", "real/a3.mp4", "fake/b1.mp4", "fake/b2.mp4"]
MISSING = {"real/a3.mp4"}
BROKEN = {"fake/b1.mp4"}


class FakeRecognizer:
    calls: list[str] = []

    class feature_config:
        rate = 16000

    def transcribe(self, wave):
        FakeRecognizer.calls.append(wave.name)
        return type("T", (), {"text": f"words of {wave.name}"})()


def fake_load_audio(path, rate):
    if Path(path).name == "b1.mp4":
        raise RuntimeError("ffmpeg failed on b1:\nline one\nline two")  # newlines inside the detail
    wave = Waveform(samples=__import__("numpy").zeros(rate // 2), rate=rate)
    wave.name = Path(path).name
    return wave


@pytest.fixture
def setup(tmp_path, monkeypatch):
    transcripts = tmp_path / "transcriptions"
    videos_dir = tmp_path / "videos"
    for video in VIDEOS:
        label, name = video.split("/")
        stem = name[:-4]
        for branch in ("whisper", "avsr"):
            d = transcripts / branch / "test" / label
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{stem}.txt").write_text("text", encoding="utf-8")
        if video not in MISSING:
            (videos_dir / label).mkdir(parents=True, exist_ok=True)
            (videos_dir / video).write_bytes(b"video")
    model = tmp_path / "model.psm"
    model.write_bytes(b"model one")
    FakeRecognizer.calls = []
    monkeypatch.setattr(purescribe.recognizer.Recognizer, "load", classmethod(lambda cls, p: FakeRecognizer()))
    monkeypatch.setattr(purescribe.audio, "load_audio", fake_load_audio)
    return tmp_path, transcripts, videos_dir, model


def run(setup, output, *extra):
    tmp_path, transcripts, videos_dir, model = setup
    return lit.main(["transcribe", "--model", str(model), "--baseline-transcripts", str(transcripts),
                     "--videos-dir", str(videos_dir), "--output", str(output), *extra])


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream, delimiter="\t"))


def order():
    # baseline_videos order: real before fake, names sorted
    return [v for v in VIDEOS if v.startswith("real/")] + [v for v in VIDEOS if v.startswith("fake/")]


def test_fresh_run_writes_every_video_and_reports_progress(setup, capsys):
    output = setup[0] / "out.tsv"
    assert run(setup, output) == 0
    rows = read_rows(output)
    assert rows[0] == lit.TSV_COLUMNS
    assert [r[0] for r in rows[1:]] == order()
    status = {r[0]: r[1] for r in rows[1:]}
    assert status["real/a3.mp4"] == "missing" and status["fake/b1.mp4"] == "error"
    assert "\n" in next(r[3] for r in rows[1:] if r[0] == "fake/b1.mp4")
    out = capsys.readouterr().out
    assert "[1/5] real/a1.mp4: extracting audio" in out
    assert "[1/5] real/a1.mp4: 0.50 s of audio, decoding" in out
    assert "[1/5] real/a1.mp4: ok in " in out and "[3/5] real/a3.mp4: missing in " in out
    lit.read_purescribe(output)  # phase 6 accepts it


def test_existing_output_is_refused_without_resume(setup, capsys):
    output = setup[0] / "out.tsv"
    output.write_text("anything\n", encoding="utf-8")
    assert run(setup, output) == 2
    assert output.read_text(encoding="utf-8") == "anything\n"
    assert "--resume" in capsys.readouterr().err
    assert FakeRecognizer.calls == []


def reference_bytes(setup):
    output = setup[0] / "reference.tsv"
    assert run(setup, output) == 0
    FakeRecognizer.calls = []
    return output.read_bytes()


def test_resume_skips_complete_rows_without_duplicates(setup, capsys):
    full = reference_bytes(setup)
    output = setup[0] / "out.tsv"
    lines = full.split(b"\n")
    output.write_bytes(b"\n".join(lines[:3]) + b"\n")  # header + 2 rows
    assert run(setup, output, "--resume") == 0
    assert output.read_bytes() == full
    assert FakeRecognizer.calls == ["b2.mp4"]  # a1, a2 kept; a3 missing, b1 broken: no recognizer call
    assert "resuming" in capsys.readouterr().out
    videos = [r[0] for r in read_rows(output)[1:]]
    assert len(videos) == len(set(videos)) == len(VIDEOS)


def test_resume_of_complete_file_does_nothing(setup):
    full = reference_bytes(setup)
    output = setup[0] / "out.tsv"
    output.write_bytes(full)
    assert run(setup, output, "--resume") == 0
    assert output.read_bytes() == full and FakeRecognizer.calls == []


@pytest.mark.parametrize("cut", ["mid_row", "before_newline", "inside_quoted_detail"])
def test_partial_last_row_is_removed_and_redone(setup, cut, capsys):
    full = reference_bytes(setup)
    text = full.decode("utf-8")
    output = setup[0] / "out.tsv"
    if cut == "inside_quoted_detail":
        position = text.index("line one") + 4  # inside the quoted multi-line field of fake/b1
    else:
        rows_end = [i + 1 for i, c in enumerate(text) if c == "\n"]
        row_end = rows_end[2]  # header + 2 complete rows end here
        position = rows_end[3] - (1 if cut == "before_newline" else 20)
        assert position > row_end
    output.write_bytes(full[: len(text[:position].encode("utf-8"))])
    assert run(setup, output, "--resume") == 0
    assert output.read_bytes() == full
    assert "removed an incomplete last row" in capsys.readouterr().out


def test_cut_off_header_restarts_from_scratch(setup):
    full = reference_bytes(setup)
    output = setup[0] / "out.tsv"
    output.write_bytes(full[:7])
    assert run(setup, output, "--resume") == 0
    assert output.read_bytes() == full


def test_resume_with_another_model_is_refused(setup):
    full = reference_bytes(setup)
    output = setup[0] / "out.tsv"
    output.write_bytes(full[: full.index(b"\n", full.index(b"\n") + 1) + 1])  # header + 1 row
    before = output.read_bytes()
    setup[3].write_bytes(b"model two")
    assert run(setup, output, "--resume") == 2
    assert output.read_bytes() == before and FakeRecognizer.calls == []


@pytest.mark.parametrize("damage", ["order", "unknown_video", "header", "fields", "videos_dir", "too_many"])
def test_inconsistent_existing_rows_are_refused(setup, damage):
    full = reference_bytes(setup)
    rows = read_rows_bytes(full)
    if damage == "order":
        rows[1], rows[2] = rows[2], rows[1]
    elif damage == "unknown_video":
        rows[1][0] = "real/zz.mp4"
    elif damage == "header":
        rows[0][2] = "text"
    elif damage == "fields":
        rows[2] = rows[2][:4]
    elif damage == "videos_dir":
        rows[1][3] = r"E:\elsewhere\real\a1.mp4"
    elif damage == "too_many":
        rows.append(rows[1])
    output = setup[0] / "out.tsv"
    write_rows(output, rows)
    before = output.read_bytes()
    assert run(setup, output, "--resume") == 2
    assert output.read_bytes() == before and FakeRecognizer.calls == []


def read_rows_bytes(data):
    import io
    return list(csv.reader(io.StringIO(data.decode("utf-8"), newline=""), delimiter="\t"))


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(rows)
