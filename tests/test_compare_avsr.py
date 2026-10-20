"""Phase 6 with an optional new Auto-AVSR export: historical mode unchanged, new texts checked.

Synthetic transcripts, TSV and manifest only; nothing is decoded and Whisper is never run.
"""

import hashlib
import json
from pathlib import Path

import pytest

import lost_in_translation as lit

# video: (whisper text, baseline lip-reading text, PureScribe status, PureScribe text)
VIDEOS = {
    "real/r1.mp4": ("the cat sat", "the cat sat", "ok", "the cat sat"),
    "real/r2.mp4": ("hello there", "hello where", "ok", "yellow"),
    "real/r3.mp4": ("gone", "gone", "missing", ""),  # MP4 no longer on disk: no manifest entry needed
    "fake/memo-1.mp4": ("buy now", "something else", "ok", "buy now"),
    "fake/diff2lip-2.mp4": ("a b c", "a b d", "ok", "x y z"),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def setup(tmp_path):
    transcripts = tmp_path / "transcriptions"
    for video, (whisper, lips, _, _) in VIDEOS.items():
        label, name = video.split("/")
        for branch, text in (("whisper", whisper), ("avsr", lips)):
            d = transcripts / branch / "test" / label
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name[:-4]}.txt").write_text(text, encoding="utf-8")
    model = tmp_path / "model.psm"
    model.write_bytes(b"model one")
    tsv = tmp_path / "purescribe.tsv"
    tsv.write_text(lit.TSV_HEADER + "".join(f"{v}\t{s}\t{t}\t\t{sha(b'model one')}\n"
                                            for v, (_, _, s, t) in VIDEOS.items()), encoding="utf-8")
    return tmp_path, transcripts, tsv, model


def export(tmp_path, texts, statuses=None, extra=()):
    """A plan-004 style output root: TXT only for status ok, manifest with byte hashes."""
    root = tmp_path / "avsr_export"
    entries = []
    for video, text in texts.items():
        status = (statuses or {}).get(video, "ok")
        label, name = video.split("/")
        entry = {"key": video, "status": status}
        if status == "ok":
            path = root / "avsr" / "test" / label / f"{name[:-4]}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
            entry.update(text_path=f"avsr/test/{label}/{name[:-4]}.txt", text_sha256=sha(text.encode("utf-8")))
        else:
            entry["reason"] = "no_face" if status == "error" else None
        entries.append(entry)
    manifest = root / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"parameters": {"subset": "test", "checkpoint": {"sha256": "c0ffee"},
                                                   "preprocessing": {"fps": 25}},
                                    "videos": entries + list(extra)}), encoding="utf-8")
    return root, manifest


def baseline_texts(skip=("real/r3.mp4",)):
    return {v: lips for v, (_, lips, _, _) in VIDEOS.items() if v not in skip}


def compare(setup, output, *extra):
    _, transcripts, tsv, _ = setup
    return lit.main(["compare", "--baseline-transcripts", str(transcripts), "--purescribe", str(tsv),
                     "--output", str(output), *extra])


def new_mode(setup, root, manifest):
    return ["--avsr-transcripts", str(root), "--avsr-manifest", str(manifest), "--model", str(setup[3])]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_historical_mode_is_unchanged(setup):
    tmp_path = setup[0]
    assert compare(setup, tmp_path / "a.json") == 0
    assert compare(setup, tmp_path / "b.json", "--model", str(setup[3])) == 0
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()  # --model only checks
    result = load(tmp_path / "a.json")
    assert result["videos"]["excluded"] == {"missing": 1} and "excluded_avsr" not in result["videos"]
    assert not any(k.startswith("avsr") for k in result["inputs"])
    assert result["protocol"]["video_branch"] == "baseline lip-reading transcripts, unchanged"
    w, p = result["whisper_baseline"], result["purescribe"]
    assert (w["tp"], w["tn"], w["fp"], w["fn"]) == (1, 2, 0, 1)
    assert (p["tp"], p["tn"], p["fp"], p["fn"]) == (2, 1, 1, 0)


def test_new_root_with_the_same_texts_gives_the_same_metrics(setup):
    tmp_path = setup[0]
    root, manifest = export(tmp_path, baseline_texts())
    assert compare(setup, tmp_path / "old.json") == 0
    assert compare(setup, tmp_path / "new.json", *new_mode(setup, root, manifest)) == 0
    old, new = load(tmp_path / "old.json"), load(tmp_path / "new.json")
    for key in ("whisper_baseline", "purescribe", "purescribe_minus_whisper", "per_video"):
        assert new[key] == old[key]
    assert new["inputs"]["baseline_texts_sha256"] == old["inputs"]["baseline_texts_sha256"]
    assert new["inputs"]["avsr_manifest_sha256"] == sha(manifest.read_bytes())
    assert new["inputs"]["avsr_checkpoint"] == {"sha256": "c0ffee"}
    assert new["videos"]["excluded_avsr"] == {} and new["videos"]["excluded"] == {"missing": 1}


def test_avsr_failures_are_excluded_for_both_systems(setup):
    tmp_path = setup[0]
    statuses = {"real/r2.mp4": "empty", "fake/memo-1.mp4": "error", "fake/diff2lip-2.mp4": "missing"}
    root, manifest = export(tmp_path, baseline_texts(), statuses)
    assert compare(setup, tmp_path / "out.json", *new_mode(setup, root, manifest)) == 0
    result = load(tmp_path / "out.json")
    assert [r["video"] for r in result["per_video"]] == ["real/r1.mp4"]
    assert result["whisper_baseline"]["tn"] == result["purescribe"]["tn"] == 1
    assert result["videos"]["excluded_avsr"] == {"empty": 1, "error": 1, "missing": 1}
    assert {"video": "fake/memo-1.mp4", "status": "error", "reason": "no_face"} in result["videos"]["excluded_avsr_videos"]


def test_ok_entry_without_its_text_fails_without_output(setup):
    tmp_path = setup[0]
    root, manifest = export(tmp_path, baseline_texts())
    (root / "avsr" / "test" / "fake" / "memo-1.txt").unlink()
    assert compare(setup, tmp_path / "out.json", *new_mode(setup, root, manifest)) == 2
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("damage", ["text_hash", "duplicate", "foreign", "incomplete", "model", "no_manifest"])
def test_inconsistent_inputs_are_refused_without_output(setup, damage):
    tmp_path = setup[0]
    texts = baseline_texts()
    extra = {"duplicate": [{"key": "real/r1.mp4", "status": "empty"}],
             "foreign": [{"key": "fake/other.mp4", "status": "empty"}]}.get(damage, ())
    if damage == "incomplete":
        texts.pop("fake/memo-1.mp4")
    root, manifest = export(tmp_path, texts, extra=extra)
    if damage == "text_hash":
        (root / "avsr" / "test" / "real" / "r2.txt").write_bytes(b"hello where\n")
    if damage == "model":
        setup[3].write_bytes(b"model two")
    args = new_mode(setup, root, manifest)
    if damage == "no_manifest":
        args = args[:2] + args[4:]
    assert compare(setup, tmp_path / "out.json", *args) == 2
    assert not (tmp_path / "out.json").exists()


def test_purescribe_missing_rows_need_no_manifest_entry(setup):
    tmp_path = setup[0]
    root, manifest = export(tmp_path, baseline_texts())   # no entry at all for real/r3
    assert compare(setup, tmp_path / "out.json", *new_mode(setup, root, manifest)) == 0
    assert load(tmp_path / "out.json")["videos"]["excluded"] == {"missing": 1}
