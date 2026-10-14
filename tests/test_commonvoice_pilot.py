import csv
from argparse import Namespace
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

import commonvoice_pilot as pilot


def record(path, sentence, speaker):
    return {"source_path": path, "audio": f"validated/{path}",
            "sentence": sentence, "client_id": speaker}


def test_reference_map_reads_utf8_csv_quoting(tmp_path):
    path = tmp_path / "references.txt"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream, delimiter="\t").writerows([
            ["a.mp3", 'He said "We’re ready."'],
            ["b.mp3", "a\tquoted field"],
        ])
    assert pilot.read_reference_map(path) == {
        "a.mp3": 'He said "We’re ready."', "b.mp3": "a\tquoted field"
    }


@pytest.mark.parametrize("content", [
    "a.mp3\tone\na.mp3\ttwo\n", "a.mp3\tone\nA.mp3\ttwo\n",
    "../a.mp3\tone\n", "C:/audio.mp3\tone\n", "a.mp3\tone\textra\n",
])
def test_reference_map_rejects_ambiguous_or_unsafe_identifiers(tmp_path, content):
    path = tmp_path / "references.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        pilot.read_reference_map(path)


def test_reference_map_filters_unrelated_rows_before_column_validation(tmp_path):
    path = tmp_path / "references.txt"
    path.write_text("a.mp3\tOfficial reference\nother.mp3\tOutside validated\t\n\nb.mp3\tSecond reference\n",
                    encoding="utf-8")
    audit = Counter()
    assert pilot.read_reference_map(path, {"a.mp3", "b.mp3"}, audit) == {
        "a.mp3": "Official reference", "b.mp3": "Second reference",
    }
    assert audit == {"records": 4, "outside_validated": 1, "blank_records": 1}


@pytest.mark.parametrize("content", [
    "a.mp3\tOfficial\textra\n", "a.mp3\tFirst\na.mp3\tConflicting\n",
    "a.mp3\tFirst\nA.mp3\tConflicting\n",
])
def test_reference_map_still_rejects_malformed_or_duplicate_selected_rows(tmp_path, content):
    path = tmp_path / "references.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        pilot.read_reference_map(path, {"a.mp3"})


def test_reference_map_resolves_windows_case_alias_to_local_basename(tmp_path):
    path = tmp_path / "references.txt"
    path.write_text("A.MP3\tOfficial reference\n", encoding="utf-8")
    assert pilot.read_reference_map(path, {"a.mp3"}) == {"a.mp3": "Official reference"}


def test_split_is_deterministic_and_keeps_speakers_together():
    records = [record(f"{speaker}-{take}.mp3", "hello", str(speaker))
               for speaker in range(30) for take in range(3)]
    train, evaluation = pilot.split_records(records, 42)
    assert (train, evaluation) == pilot.split_records(list(reversed(records)), 42)
    assert train and evaluation
    assert not {r["client_id"] for r in train} & {r["client_id"] for r in evaluation}
    assert not {r["audio"] for r in train} & {r["audio"] for r in evaluation}
    assert len(train) + len(evaluation) == len(records)


def test_vocabulary_uses_training_support_and_preserves_contractions():
    training = [record(f"yes{i}.mp3", "YES!", str(i)) for i in range(6)]
    training += [record(f"no{i}.mp3", "no", str(i)) for i in range(5)]
    training += [record(f"contract{i}.mp3", "we're", str(i)) for i in range(20)]
    vocabulary, availability = pilot.choose_vocabulary(training, 2, 3)
    assert vocabulary == ["yes", "no"]
    assert availability["yes"] == {
        "training_occurrences": 6, "isolated_recordings": 6, "available_speakers": 6,
    }
    assert "were" not in availability


def test_training_schedule_is_bounded_unique_and_includes_connected_speech():
    training = [record(f"yes{i}.mp3", "yes", str(i)) for i in range(8)]
    training += [record(f"both{i}.mp3", "yes no", str(i + 10)) for i in range(6)]
    selected = pilot.training_schedule(training, ["yes", "no"], 42, 7)
    assert len(selected) == 7
    assert len({r["audio"] for r in selected}) == 7
    assert any(r["sentence"] == "yes no" for r in selected)
    assert selected == pilot.training_schedule(list(reversed(training)), ["yes", "no"], 42, 7)


def test_extraction_uses_official_label_and_only_matched_whisper_bounds(tmp_path):
    corpus = tmp_path / "word_corpus"
    corpus.mkdir()
    words = [SimpleNamespace(word="were", start=0.0, end=0.1),
             SimpleNamespace(word=" about", start=0.2, end=0.5),
             SimpleNamespace(word="extra", start=0.7, end=0.9)]
    counts, speakers = Counter(), defaultdict(Counter)
    outcomes, obtained, matched = pilot.extract_record(
        record("a.mp3", "We’re ABOUT, brown", "speaker"), np.ones(16000) * 0.25, words,
        {"were", "about", "brown"}, corpus, counts, speakers,
    )
    assert [item["status"] for item in outcomes] == ["unmatched", "saved", "unmatched"]
    assert obtained == 3 and matched == 1
    accepted = outcomes[1]
    assert accepted["official_word"] == "ABOUT," and accepted["label"] == "about"
    assert accepted["whisper_start"] == 0.2 and accepted["whisper_end"] == 0.5
    assert accepted["clip_start"] == 0.15 and accepted["clip_end"] == 0.62
    assert list(corpus.rglob("*.wav")) == [corpus / "about" / "about_0000.wav"]
    assert counts == {"about": 1}


def test_extraction_rejects_bad_timestamps_and_existing_word_file(tmp_path):
    corpus = tmp_path / "word_corpus"
    corpus.mkdir()
    (corpus / "about").write_bytes(b"preserve this file")
    words = [SimpleNamespace(word="about", start=float("nan"), end=0.5),
             SimpleNamespace(word="about", start=0.2, end=0.5)]
    outcomes, _, _ = pilot.extract_record(
        record("a.mp3", "about about", "speaker"), np.ones(16000) * 0.25, words,
        {"about"}, corpus, Counter(), defaultdict(Counter),
    )
    assert [item["status"] for item in outcomes] == ["invalid_timestamp", "save_error"]
    assert (corpus / "about").read_bytes() == b"preserve this file"


def test_extraction_counts_existing_clips_and_never_overwrites(tmp_path):
    corpus = tmp_path / "word_corpus"
    (corpus / "about").mkdir(parents=True)
    original = corpus / "about" / "about_0000.wav"
    original.write_bytes(b"existing")
    words = [SimpleNamespace(word="about", start=0.2, end=0.5)]
    counts = Counter(about=1)
    arguments = (record("a.mp3", "about", "speaker"), np.ones(16000) * 0.25, words,
                 {"about"}, corpus, counts, defaultdict(Counter))
    outcomes, _, _ = pilot.extract_record(*arguments, cap=2)
    assert outcomes[0]["clip"].endswith("about_0001.wav")
    outcomes, _, _ = pilot.extract_record(*arguments, cap=2)
    assert outcomes[0]["status"] == "word_limit"
    assert counts["about"] == 2 and original.read_bytes() == b"existing"


def test_boundary_diagnostics_flag_neighbor_overlap():
    words = [SimpleNamespace(word="first", start=0.0, end=0.2),
             SimpleNamespace(word="about", start=0.2, end=0.4),
             SimpleNamespace(word="last", start=0.4, end=0.6)]
    metrics = pilot.boundary_metrics(np.ones(16000), 2400, 8320, 1, words)
    assert metrics["previous_word_overlap_seconds"] == pytest.approx(0.05)
    assert metrics["next_word_overlap_seconds"] == pytest.approx(0.12)
    assert metrics["start_rms_ratio"] == 1.0


def test_extraction_rejects_timestamp_pointing_to_leading_silence(tmp_path):
    corpus = tmp_path / "word_corpus"
    corpus.mkdir()
    audio = np.concatenate([np.zeros(8000), np.ones(8000) * 0.25])
    words = [SimpleNamespace(word="yes", start=0.0, end=0.1)]
    counts = Counter()
    outcomes, _, _ = pilot.extract_record(
        record("a.mp3", "yes", "speaker"), audio, words, {"yes"}, corpus,
        counts, defaultdict(Counter),
    )
    assert outcomes[0]["status"] == "low_signal"
    assert counts["yes"] == 0 and not list(corpus.rglob("*.wav"))


def test_training_staging_excludes_low_support_without_duplicating_classes(tmp_path):
    for word, count in (("yes", 2), ("hey", 2), ("five", 1)):
        directory = tmp_path / "word_corpus" / word
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"{word}_{index:04d}.wav").write_bytes(f"{word}{index}".encode())
    per_word = {word: {"accepted": count} for word, count in (("yes", 2), ("hey", 2), ("five", 1))}
    stage = getattr(pilot, "stage_training_corpus", None)
    assert callable(stage), "training must enforce its minimum using training-only evidence"
    selection = stage(tmp_path, per_word, 2)
    assert selection["vocabulary"] == ["hey", "yes"]
    assert selection["excluded_low_support"] == ["five"]
    assert len(list((tmp_path / "training_corpus").rglob("*.wav"))) == 4
    assert (tmp_path / "word_corpus" / "five" / "five_0000.wav").read_bytes() == b"five0"
    assert (tmp_path / "training_corpus" / "yes" / "yes_0000.wav").samefile(
        tmp_path / "word_corpus" / "yes" / "yes_0000.wav")
    with pytest.raises(FileExistsError):
        stage(tmp_path, per_word, 2)


def test_scoring_reports_oov_full_and_known_subset_without_training_scores():
    references = {"new/a.mp3": "Hello, about!", "new/b.mp3": "strange about",
                  "new/c.mp3": "about"}
    predictions = {"new/a.mp3": "hello about", "new/b.mp3": "hello about extra"}
    metrics = pilot.score_predictions(references, predictions, {"hello", "about"})
    assert metrics["all"] == {
        "sentences": 3, "reference_words": 5, "substitutions": 1, "deletions": 1,
        "insertions": 1, "errors": 3, "correct_sentences": 1,
        "wer_percent": 60.0, "sentence_accuracy_percent": pytest.approx(100 / 3),
    }
    assert metrics["in_vocabulary"]["sentences"] == 2
    assert metrics["in_vocabulary"]["wer_percent"] == pytest.approx(100 / 3)
    assert metrics["oov_words"] == {"strange": 1}


def test_scoring_preserves_apostrophes_and_handles_empty_known_subset():
    metrics = pilot.score_predictions({"a.mp3": "We’re"}, {"a.mp3": "were"}, {"were"})
    assert metrics["all"]["substitutions"] == 1
    assert metrics["in_vocabulary"]["sentences"] == 0
    assert metrics["in_vocabulary"]["wer_percent"] is None
    assert metrics["in_vocabulary"]["sentence_accuracy_percent"] is None


def test_json_stream_supports_multiple_cli_json_objects():
    assert pilot.json_stream(' {"audio":"one.mp3","text":"hello"}\n{"audio":"two.mp3","text":""}\n') == [
        {"audio": "one.mp3", "text": "hello"}, {"audio": "two.mp3", "text": ""},
    ]


def test_prepare_keeps_raw_metadata_quotes_and_refuses_output_reuse(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "validated").mkdir(parents=True)
    rows = []
    for label in ("yes", "no"):
        speakers = [str(i) for i in range(100)
                    if int.from_bytes(pilot.rank(42, str(i))[:8], "big") / 2**64 >= 0.2][:8]
        rows += [(f"{label}{speaker}.mp3", label, speaker) for speaker in speakers]
    held_out = next(str(i) for i in range(100)
                    if int.from_bytes(pilot.rank(42, str(i))[:8], "big") / 2**64 < 0.2)
    rows += [("eval.mp3", '"Quoted evaluation', held_out)]
    outside = ("outside.mp3", "yes", speakers[0])
    with (source / "refs.txt").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t").writerows((name, sentence) for name, sentence, _ in [*rows, outside])
    metadata = "client_id\tpath\tsentence\tlocale\n" + "".join(
        f"{speaker}\t{name.upper() if name == rows[0][0] else name}\t{sentence}\ten\n"
        for name, sentence, speaker in [*rows, outside])
    (source / "metadata.tsv").write_text(metadata, encoding="utf-8")
    for name, _, _ in rows:
        (source / "validated" / name).write_bytes(name.encode())
    (source / "validated" / "unmapped.mp3").write_bytes(b"no reference")
    (source / "validated" / "notes.txt").write_text("not audio", encoding="utf-8")
    (source / "clips").mkdir()
    (source / "clips" / outside[0]).write_bytes(b"must never be used")
    monkeypatch.setattr(pilot.importlib.metadata, "version", lambda name: "test")
    args = Namespace(data_root=source, output=tmp_path / "experiment", reference="refs.txt",
                     metadata="metadata.tsv", seed=42, min_clips=2, goal_clips=3,
                     vocabulary_size=2, max_train=12, max_evaluation=2)
    pilot.prepare(args)
    import json
    plan = json.loads((args.output / "plan.json").read_text(encoding="utf-8"))
    assert plan["usable_rows"] == len(rows)
    assert plan["source_rows"] == len(rows) + 1
    assert plan["validated_audio_files"] == len(rows) + 1
    assert plan["validated_audio_without_reference"] == ["unmapped.mp3"]
    assert plan["reference_audit"]["outside_validated"] == 1
    assert all(r["audio"].startswith("validated/")
               for r in [*plan["training_selected"], *plan["evaluation_selected"]])
    assert outside[0] not in {r["source_path"] for r in plan["training_selected"]}
    assert plan["speaker_overlap"] == 0
    assert plan["source_metadata_text_disagreements"] == []
    assert plan["evaluation_selected"][0]["sentence"] == '"Quoted evaluation'
    assert not {r["sha256"] for r in plan["training_selected"]} & {r["sha256"] for r in plan["evaluation_selected"]}
    with pytest.raises(FileExistsError):
        pilot.prepare(args)


def test_evaluation_membership_is_independent_of_vocabulary_with_duplicate_audio(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "validated").mkdir(parents=True)
    training_speakers = [str(i) for i in range(100)
                         if int.from_bytes(pilot.rank(42, str(i))[:8], "big") / 2**64 >= 0.2][:8]
    held_out = next(str(i) for i in range(100)
                    if int.from_bytes(pilot.rank(42, str(i))[:8], "big") / 2**64 < 0.2)
    rows = [(f"{word}{speaker}.mp3", word, speaker)
            for word in ("yes", "no", "zzz") for speaker in training_speakers]
    rows += [("eval_unique.mp3", "held out", held_out),
             ("eval_duplicate.mp3", "duplicate held out", held_out)]
    with (source / "refs.txt").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t").writerows((name, text) for name, text, _ in rows)
    (source / "metadata.tsv").write_text("client_id\tpath\tsentence\tlocale\n" + "".join(
        f"{speaker}\t{name}\t{text}\ten\n" for name, text, speaker in rows), encoding="utf-8")
    for name, _, _ in rows:
        (source / "validated" / name).write_bytes(name.encode())
    duplicate = min((name for name, text, _ in rows if text == "zzz"), key=lambda name: pilot.rank(42, name))
    (source / "validated" / "eval_duplicate.mp3").write_bytes(duplicate.encode())
    monkeypatch.setattr(pilot.importlib.metadata, "version", lambda name: "test")
    plans = []
    import json
    for size in (2, 3):
        args = Namespace(data_root=source, output=tmp_path / f"experiment{size}", reference="refs.txt",
                         metadata="metadata.tsv", seed=42, min_clips=2, goal_clips=3,
                         vocabulary_size=size, max_train=12, max_evaluation=2)
        pilot.prepare(args)
        plans.append(json.loads((args.output / "plan.json").read_text(encoding="utf-8")))
    assert [[r["audio"] for r in plan["evaluation_selected"]] for plan in plans] == [
        ["validated/eval_unique.mp3", "validated/eval_duplicate.mp3"],
        ["validated/eval_unique.mp3", "validated/eval_duplicate.mp3"],
    ]
    assert duplicate not in {r["source_path"] for r in plans[1]["training_selected"]}
