"""Bounded, speaker-separated Common Voice experiment; artifacts live outside the repo.

Run prepare, extract, then evaluate with the same --output directory.
The headerless reference map is CSV-quoted TSV; CV metadata is unquoted TSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

from evaluate_sequences import levenshtein
from script import (
    CLIPS_PER_WORD, MIN_CLIP_DURATION_SEC, PADDING_END_SEC, PADDING_START_SEC,
    TARGET_SR, _comparison_word, clean_word, ensure_word_directory,
    match_word_indices, save_wav, valid_word_bounds,
)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_reference_map(path):
    references = {}
    seen = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.reader(stream, delimiter="\t", strict=True)):
            if index == 0 and row == ["path", "sentence"]:
                continue
            if len(row) != 2:
                raise ValueError(f"Reference record {index + 1}: expected two fields")
            name, sentence = row
            relative = PureWindowsPath(name)
            if not name or relative.drive or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe audio identifier: {name!r}")
            if name.casefold() in seen:
                raise ValueError(f"Duplicate audio identifier: {name}")
            seen.add(name.casefold())
            references[name] = sentence
    return references


def tokens(sentence):
    return [word for word in sentence.split() if clean_word(word)]


def normalized_tokens(sentence):
    return [_comparison_word(word) for word in tokens(sentence)]


def rank(seed, key):
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()


def split_records(records, seed, evaluation_fraction=0.2):
    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation_fraction must be between zero and one")
    training, evaluation = [], []
    for record in sorted(records, key=lambda item: item["source_path"]):
        group = record["client_id"] or record["source_path"]
        held_out = int.from_bytes(rank(seed, group)[:8], "big") / 2**64 < evaluation_fraction
        (evaluation if held_out else training).append(record)
    return training, evaluation


def choose_vocabulary(training, size, min_candidates, min_speakers=5):
    occurrences, isolated = Counter(), Counter()
    speakers = defaultdict(set)
    for record in training:
        words = normalized_tokens(record["sentence"])
        for word in words:
            if word.isascii() and word.isalpha() and word not in {"con", "prn", "aux", "nul"}:
                occurrences[word] += 1
                if record["client_id"]:
                    speakers[word].add(record["client_id"])
                if len(words) == 1:
                    isolated[word] += 1
    eligible = [word for word in isolated
                if isolated[word] >= min_candidates and len(speakers[word]) >= min_speakers]
    vocabulary = sorted(eligible, key=lambda word: (-isolated[word], word))[:size]
    availability = {
        word: {"training_occurrences": occurrences[word], "isolated_recordings": isolated[word],
               "available_speakers": len(speakers[word])}
        for word in vocabulary
    }
    return vocabulary, availability


def training_schedule(training, vocabulary, seed, limit):
    buckets = {word: [[], []] for word in vocabulary}
    for record in training:
        words = tokens(record["sentence"])
        if len(words) > 24:
            continue
        for word in {clean_word(token) for token in words} & set(vocabulary):
            buckets[word][0 if len(words) == 1 else 1].append(record)
    for pair in buckets.values():
        for bucket in pair:
            bucket.sort(key=lambda item: rank(seed, item["source_path"]))
    positions = {word: [0, 0] for word in vocabulary}
    selected, seen, round_index = [], set(), 0
    while len(selected) < limit:
        advanced = False
        for word in vocabulary:
            order = [1, 0] if round_index % 3 == 2 else [0, 1]
            for bucket_index in order:
                position = positions[word][bucket_index]
                bucket = buckets[word][bucket_index]
                if position < len(bucket):
                    record = bucket[position]
                    positions[word][bucket_index] += 1
                    advanced = True
                    if record["source_path"] not in seen:
                        seen.add(record["source_path"])
                        selected.append(record)
                    break
            if len(selected) >= limit:
                break
        if not advanced:
            break
        round_index += 1
    return selected


def prepare(args):
    data_root, output = args.data_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW experiment directory: {output}")
    reference = data_root / args.reference
    references = read_reference_map(reference)
    metadata_path = data_root / args.metadata
    metadata, disagreements = {}, []
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        if not {"client_id", "path", "sentence"}.issubset(reader.fieldnames or []):
            raise ValueError("Metadata requires client_id, path, sentence")
        for row in reader:
            if row["path"] in references:
                if row["path"] in metadata:
                    raise ValueError(f"Ambiguous metadata for {row['path']}")
                metadata[row["path"]] = {key: row.get(key, "") for key in ("client_id", "locale")}
                if row["sentence"] != references[row["path"]]:
                    disagreements.append(row["path"])
    usable, rejected = [], []
    for name, sentence in references.items():
        reason = None
        if not tokens(sentence) or any(char in sentence for char in "\r\n\t"):
            reason = "empty_or_multiline_reference"
        elif name not in metadata:
            reason = "missing_speaker_metadata"
        elif metadata[name]["locale"] not in {"", "en"}:
            reason = "non_english"
        audio = next((data_root / directory / name for directory in ("validated", "clips")
                      if (data_root / directory / name).is_file()), None)
        if audio is None:
            reason = "missing_audio"
        if reason:
            rejected.append({"source_path": name, "reason": reason})
        else:
            usable.append({"source_path": name, "audio": audio.relative_to(data_root).as_posix(),
                           "sentence": sentence, "client_id": metadata[name]["client_id"]})
    training, held_out = split_records(usable, args.seed)
    vocabulary, availability = choose_vocabulary(
        training, args.vocabulary_size, math.ceil(args.min_clips * 1.5)
    )
    if len(vocabulary) < 2:
        raise ValueError("Insufficient training-only vocabulary support")
    # Freeze evaluation before removing duplicate training audio, independently of vocabulary.
    evaluation_selected, eval_hashes = [], set()
    for record in sorted(held_out, key=lambda item: rank(args.seed, "eval:" + item["source_path"])):
        fingerprint = hashlib.sha256((data_root / record["audio"]).read_bytes()).hexdigest()
        if fingerprint in eval_hashes:
            continue
        eval_hashes.add(fingerprint)
        evaluation_selected.append({**record, "sha256": fingerprint})
        if len(evaluation_selected) >= args.max_evaluation:
            break
    schedule = training_schedule(training, vocabulary, args.seed, args.max_train)
    train_hashes, training_selected = set(), []
    for record in schedule:
        fingerprint = hashlib.sha256((data_root / record["audio"]).read_bytes()).hexdigest()
        if fingerprint not in train_hashes and fingerprint not in eval_hashes:
            train_hashes.add(fingerprint)
            training_selected.append({**record, "sha256": fingerprint})
    train_speakers = {record["client_id"] for record in training if record["client_id"]}
    eval_speakers = {record["client_id"] for record in held_out if record["client_id"]}
    assert not train_speakers & eval_speakers
    assert not train_hashes & eval_hashes
    plan = {
        "data_root": str(data_root), "reference_file": str(reference),
        "reference_format": "UTF-8 CSV-quoted TSV, two fields path/sentence, no header",
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "metadata_file": str(metadata_path), "metadata_columns": ["path", "client_id", "locale"],
        "metadata_format": "UTF-8 unquoted TSV with header; sentence audited, not substituted",
        "source_rows": len(references), "usable_rows": len(usable), "rejected": rejected,
        "source_metadata_text_disagreements": disagreements,
        "seed": args.seed, "split": "SHA256(seed:client_id), first 64 bits / 2^64 < 0.2 -> evaluation; missing client_id -> audio ID",
        "training_rows": len(training), "held_out_rows": len(held_out),
        "training_speakers": len(train_speakers), "held_out_speakers": len(eval_speakers),
        "speaker_overlap": 0, "missing_client_ids": sum(not r["client_id"] for r in usable),
        "vocabulary": vocabulary, "availability": availability,
        "vocabulary_policy": "top training-only isolated-word counts; ASCII letters; >=1.5*min_clips isolated samples and >=5 speakers",
        "training_qualification": "only classes with >=min_clips accepted training examples enter the model; no evaluation-guided changes",
        "min_clips": args.min_clips, "goal_clips": args.goal_clips,
        "clips_per_word_limit": CLIPS_PER_WORD, "max_training_audio": args.max_train,
        "max_evaluation_audio": args.max_evaluation,
        "training_selection": "seeded round robin per target: two isolated recordings, then one connected; <=24 reference tokens",
        "evaluation_selection": "seeded audio-ID hash order; independent of vocabulary and decoding results",
        "max_audio_seconds": 15, "max_clips_per_speaker_per_word": 2,
        "whisper_model": "base", "whisper_language": "en", "cpu_threads": 4,
        "whisper_suppress_silence": False,
        "minimum_crop_to_source_peak_rms_ratio": 0.05,
        "audio_format": {"rate": TARGET_SR, "channels": 1, "pcm_bits": 16,
                         "padding_start": PADDING_START_SEC, "padding_end": PADDING_END_SEC,
                         "minimum_clip_seconds": MIN_CLIP_DURATION_SEC},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "stable-ts", "torch", "librosa")},
        "python": sys.version, "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_selected": training_selected, "evaluation_selected": evaluation_selected,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    with (output / "partitions.tsv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["audio", "client_id", "partition"])
        for partition, records in (("training", training), ("evaluation", held_out)):
            writer.writerows((r["audio"], r["client_id"], partition) for r in records)
    with (output / "reference.tsv").open("x", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(
            (record["audio"], record["sentence"]) for record in evaluation_selected
        )
    print(json.dumps({key: plan[key] for key in ("source_rows", "usable_rows", "training_rows", "held_out_rows", "vocabulary", "availability")}, indent=2))
    print(f"Frozen plan: {output / 'plan.json'}", flush=True)


def boundary_metrics(audio, start_sample, end_sample, word_index, words):
    import numpy as np

    clip = audio[start_sample:end_sample]
    window = TARGET_SR // 100
    blocks = clip[:len(clip) // window * window].reshape(-1, window)
    peak_rms = float(np.max(np.sqrt(np.mean(blocks ** 2, axis=1)))) if len(blocks) else 0.0
    source_blocks = audio[:len(audio) // window * window].reshape(-1, window)
    source_peak_rms = float(np.max(np.sqrt(np.mean(source_blocks ** 2, axis=1)))) if len(source_blocks) else 0.0
    result = {"start_rms_ratio": 0.0, "end_rms_ratio": 0.0,
              "previous_word_overlap_seconds": 0.0, "next_word_overlap_seconds": 0.0,
              "crop_to_source_peak_rms_ratio": peak_rms / source_peak_rms if source_peak_rms else 0.0}
    if peak_rms:
        result["start_rms_ratio"] = float(np.sqrt(np.mean(clip[:window] ** 2))) / peak_rms
        result["end_rms_ratio"] = float(np.sqrt(np.mean(clip[-window:] ** 2))) / peak_rms
    for key, index in (("previous_word", word_index - 1), ("next_word", word_index + 1)):
        if 0 <= index < len(words):
            bounds = valid_word_bounds(getattr(words[index], "start", None),
                                       getattr(words[index], "end", None), len(audio) / TARGET_SR)
            if bounds:
                result[key + "_overlap_seconds"] = max(
                    0.0, min(end_sample / TARGET_SR, bounds[1]) - max(start_sample / TARGET_SR, bounds[0])
                )
                result[key] = words[index].word
    return result


def extract_record(record, audio, words, vocabulary, corpus, counts, speaker_counts,
                   cap=CLIPS_PER_WORD, speaker_cap=2):
    reference = tokens(record["sentence"])
    words = [word for word in words if isinstance(word.word, str) and clean_word(word.word)]
    matches = dict(match_word_indices(reference, [word.word for word in words]))
    outcomes = []
    for index, official_word in enumerate(reference):
        label = clean_word(official_word)
        if label not in vocabulary:
            continue
        item = {"audio": record["audio"], "client_id": record["client_id"],
                "official_word": official_word, "label": label, "reference_index": index,
                "matched": index in matches}
        outcomes.append(item)
        if index not in matches:
            item["status"] = "unmatched"
            continue
        whisper_index = matches[index]
        word = words[whisper_index]
        item["whisper_index"] = whisper_index
        bounds = valid_word_bounds(getattr(word, "start", None), getattr(word, "end", None),
                                   len(audio) / TARGET_SR)
        if bounds is None:
            item["status"] = "invalid_timestamp"
            continue
        if counts[label] >= cap:
            item["status"] = "word_limit"
            continue
        speaker = record["client_id"] or record["audio"]
        if speaker_counts[label][speaker] >= speaker_cap:
            item["status"] = "speaker_limit"
            continue
        start, end = bounds
        start_sample = int(max(0.0, start - PADDING_START_SEC) * TARGET_SR)
        end_sample = min(len(audio), int((end + PADDING_END_SEC) * TARGET_SR))
        if end_sample - start_sample < int(MIN_CLIP_DURATION_SEC * TARGET_SR):
            item["status"] = "short_clip"
            continue
        quality = boundary_metrics(audio, start_sample, end_sample, whisper_index, words)
        item["boundary_metrics"] = quality
        if quality["crop_to_source_peak_rms_ratio"] < 0.05:
            item["status"] = "low_signal"
            continue
        word_dir = corpus / label
        try:
            ensure_word_directory(word_dir)
            clip_index = 0
            while True:
                path = word_dir / f"{label}_{clip_index:04d}.wav"
                try:
                    save_wav(path, audio[start_sample:end_sample], TARGET_SR)
                    break
                except FileExistsError:
                    if not path.is_file():
                        raise
                    clip_index += 1
        except Exception as exc:
            item.update(status="save_error", error=f"{type(exc).__name__}: {exc}")
            continue
        counts[label] += 1
        speaker_counts[label][speaker] += 1
        item.update(status="saved", clip=path.relative_to(corpus.parent).as_posix(),
                    whisper_start=start, whisper_end=end,
                    clip_start=start_sample / TARGET_SR, clip_end=end_sample / TARGET_SR,
                    reference_token_count=len(reference))
    return outcomes, len(words), sum(item["matched"] for item in outcomes)


def finite_time(value):
    try:
        return float(value) if not isinstance(value, bool) and math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def extract_training(output):
    import librosa
    import stable_whisper
    import torch

    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    corpus = output / "word_corpus"
    corpus.mkdir(exist_ok=False)
    torch.set_num_threads(plan["cpu_threads"])
    torch.manual_seed(plan["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = stable_whisper.load_model(plan["whisper_model"], device=device)
    counts, speakers, stats = Counter(), defaultdict(Counter), Counter()
    ledger, started = [], time.monotonic()
    with (output / "clips.jsonl").open("x", encoding="utf-8") as clips_log, \
            (output / "whisper_training.jsonl").open("x", encoding="utf-8") as whisper_log:
        for record in plan["training_selected"]:
            if all(counts[word] >= plan["goal_clips"] for word in plan["vocabulary"]):
                break
            stats["manifest_rows_examined"] += 1
            status, error, words = None, None, []
            path = Path(plan["data_root"]) / record["audio"]
            try:
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError("Source audio changed since partitioning")
                audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
                assert sr == TARGET_SR
                if len(audio) / sr > plan["max_audio_seconds"]:
                    status = "duration_limit"
            except Exception as exc:
                status, error = "load_error", f"{type(exc).__name__}: {exc}"
            if status is None:
                try:
                    result = model.transcribe(audio, language="en", word_timestamps=True,
                                              fp16=device == "cuda", verbose=None,
                                              suppress_silence=plan["whisper_suppress_silence"])
                    words = list(result.all_words())
                except Exception as exc:
                    status, error = "whisper_error", f"{type(exc).__name__}: {exc}"
            if status:
                stats[status] += 1
                outcomes = [{"audio": record["audio"], "client_id": record["client_id"],
                             "official_word": word, "label": clean_word(word),
                             "status": status, "matched": False, "error": error}
                            for word in tokens(record["sentence"]) if clean_word(word) in plan["vocabulary"]]
                print(f"[!] {record['source_path']}: {status} {error or ''}", flush=True)
            else:
                outcomes, obtained, matched = extract_record(
                    record, audio, words, set(plan["vocabulary"]), corpus, counts, speakers,
                    cap=min(plan["goal_clips"], CLIPS_PER_WORD),
                    speaker_cap=plan["max_clips_per_speaker_per_word"],
                )
                stats["whisper_words_obtained"] += obtained
                stats["exact_matches"] += matched
                stats["whisper_words_ignored"] += obtained - matched
            stats["commonvoice_words_considered"] += len(outcomes)
            stats["reference_words_discarded"] += sum(not item["matched"] for item in outcomes)
            for item in outcomes:
                if item["status"] == "save_error":
                    print(f"[!] {record['source_path']}: {item['error']}", flush=True)
                clips_log.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            whisper_log.write(json.dumps({"audio": record["audio"], "official_sentence": record["sentence"],
                "status": status or "transcribed", "words": [
                    {"word": word.word, "start": finite_time(getattr(word, "start", None)),
                     "end": finite_time(getattr(word, "end", None))} for word in words
                ]}, ensure_ascii=False, allow_nan=False) + "\n")
            clips_log.flush()
            whisper_log.flush()
            ledger.extend(outcomes)
            print(f"[{stats['manifest_rows_examined']}/{len(plan['training_selected'])}] "
                  f"{record['source_path']} saved={sum(counts.values())} {dict(counts)}", flush=True)
    per_word = {}
    for word in plan["vocabulary"]:
        candidates = [item for item in ledger if item["label"] == word]
        reasons = Counter(item["status"] for item in candidates if item["status"] != "saved")
        per_word[word] = {**plan["availability"][word], "candidates_examined": len(candidates),
                          "accepted": counts[word], "discarded": sum(reasons.values()),
                          "discard_reasons": dict(reasons), "accepted_speakers": len(speakers[word])}
    saved = [item for item in ledger if item["status"] == "saved"]
    samples = []
    for word in plan["vocabulary"]:
        word_clips = sorted((item for item in saved if item["label"] == word),
                            key=lambda item: (item["reference_token_count"] == 1, item["clip"]))
        samples.extend(word_clips[:2])
    qa = {"method": "Reject nearly silent crops (peak 10 ms RMS <5% of source peak); padded-edge RMS and neighboring Whisper interval overlap; contextual review samples",
          "edge_rms_flag_threshold": 0.25, "neighbor_overlap_flag_seconds": 0.03,
          "edge_rms_flags": sum(max(item["boundary_metrics"]["start_rms_ratio"], item["boundary_metrics"]["end_rms_ratio"]) > 0.25 for item in saved),
          "neighbor_overlap_flags": sum(max(item["boundary_metrics"]["previous_word_overlap_seconds"], item["boundary_metrics"]["next_word_overlap_seconds"]) > 0.03 for item in saved),
          "samples": samples,
          "limitation": "Energy and estimated neighboring intervals flag risks; they do not certify acoustic/phonetic boundaries."}
    write_json(output / "qa.json", qa)
    with (output / "qa.html").open("x", encoding="utf-8") as stream:
        stream.write('<!doctype html><meta charset="utf-8"><title>Common Voice clip audit</title><h1>Training clips and source context</h1>')
        for item in samples:
            source = (Path(plan["data_root"]) / item["audio"]).as_uri()
            stream.write(f'<h2>{html.escape(item["official_word"])}</h2><p>{html.escape(item["audio"])}: '
                         f'{item["whisper_start"]:.3f}–{item["whisper_end"]:.3f}; crop '
                         f'{item["clip_start"]:.3f}–{item["clip_end"]:.3f}</p>'
                         f'<p>Crop <audio controls src="{html.escape(item["clip"])}"></audio></p>'
                         f'<p>Source <audio controls src="{html.escape(source)}#t={max(0,item["clip_start"]-0.4):.3f},{item["clip_end"]+0.4:.3f}"></audio></p>')
    stats["clips_saved"] = len(saved)
    report = {"device": device, "elapsed_seconds": time.monotonic() - started,
              "stats": dict(stats), "per_word": per_word,
              "coverage_percent": 100 * stats["exact_matches"] / stats["commonvoice_words_considered"] if stats["commonvoice_words_considered"] else 0.0}
    write_json(output / "extraction.json", report)
    print(json.dumps(report, indent=2), flush=True)


def json_stream(text):
    decoder, values, position = json.JSONDecoder(), [], 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        value, position = decoder.raw_decode(text, position)
        values.append(value)
    return values


def score_predictions(references, predictions, vocabulary):
    total, known = Counter(), Counter()
    oov, details = Counter(), []
    for audio, sentence in references.items():
        reference = normalized_tokens(sentence)
        hypothesis = normalized_tokens(predictions.get(audio, ""))
        errors, substitutions, deletions, insertions = levenshtein(reference, hypothesis)
        counts = {"sentences": 1, "reference_words": len(reference), "substitutions": substitutions,
                  "deletions": deletions, "insertions": insertions, "errors": errors,
                  "correct_sentences": int(reference == hypothesis)}
        in_vocabulary = all(word in vocabulary for word in reference)
        total.update(counts)
        if in_vocabulary:
            known.update(counts)
        oov.update(word for word in reference if word not in vocabulary)
        details.append({"audio": audio, "reference": reference, "hypothesis": hypothesis,
                        "in_vocabulary": in_vocabulary, **counts})
    def finish(counts):
        result = {key: counts[key] for key in ("sentences", "reference_words", "substitutions",
                   "deletions", "insertions", "errors", "correct_sentences")}
        result["wer_percent"] = 100 * counts["errors"] / counts["reference_words"] if counts["reference_words"] else None
        result["sentence_accuracy_percent"] = 100 * counts["correct_sentences"] / counts["sentences"] if counts["sentences"] else None
        return result
    return {"all": finish(total), "in_vocabulary": finish(known),
            "oov_occurrences": sum(oov.values()), "oov_words": dict(oov.most_common()),
            "normalization": "casefold, typographic apostrophes -> ASCII, external punctuation only",
            "per_audio": details}


def stage_training_corpus(output, per_word, minimum):
    vocabulary = sorted(word for word, row in per_word.items() if row["accepted"] >= minimum)
    if len(vocabulary) < 2:
        raise RuntimeError("Fewer than two classes meet the training-only minimum")
    corpus = output / "training_corpus"
    corpus.mkdir(exist_ok=False)
    mapping = []
    for word in vocabulary:
        sources = sorted((output / "word_corpus" / word).glob("*.wav"))
        if len(sources) != per_word[word]["accepted"]:
            raise ValueError(f"Clip inventory changed for {word}")
        (corpus / word).mkdir()
        for source in sources:
            destination = corpus / word / source.name
            # One alias per distinct accepted example, never copies to fill a class.
            os.link(source, destination)
            mapping.append({"source": source.relative_to(output).as_posix(),
                            "training": destination.relative_to(output).as_posix()})
    selection = {"vocabulary": vocabulary,
                 "excluded_low_support": sorted(set(per_word) - set(vocabulary)),
                 "minimum_accepted_clips": minimum, "clip_aliases": mapping}
    write_json(output / "training_selection.json", selection)
    return selection


def train_and_evaluate(output):
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    extraction = json.loads((output / "extraction.json").read_text(encoding="utf-8"))
    model_path = output / "commonvoice_pilot.psm"
    if model_path.exists() or (output / "predictions.jsonl").exists():
        raise FileExistsError("Model or predictions already exist; use a new experiment directory")
    selection = stage_training_corpus(output, extraction["per_word"], plan["min_clips"])
    print(f"Training-only qualified vocabulary: {selection['vocabulary']}; "
          f"low-support classes retained outside training: {selection['excluded_low_support']}", flush=True)
    commands = []
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    def invoke(arguments):
        command = [sys.executable, "-B", "-m", "purescribe.cli", *arguments]
        started = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", env=environment, timeout=1800)
        commands.append({"argv": command, "returncode": result.returncode,
                         "elapsed_seconds": time.monotonic() - started, "stderr": result.stderr})
        return result
    trained = invoke(["train", str(output / "training_corpus"), "--output", str(model_path)])
    with (output / "train.log").open("x", encoding="utf-8") as stream:
        stream.write(trained.stdout + trained.stderr)
    if trained.returncode:
        write_json(output / "commands.json", commands)
        raise RuntimeError(f"PureScribe train failed: {trained.stderr}")
    print(trained.stdout, flush=True)
    from purescribe.recognizer import Recognizer
    vocabulary = set(Recognizer.load(model_path).vocabulary)
    predictions, failures = {}, {}
    data_root = Path(plan["data_root"])
    # Only audio paths reach the CLI. Ground truth is read for scoring AFTER decoding.
    records = plan["evaluation_selected"]
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for offset in range(0, len(records), 32):
            chunk = records[offset:offset + 32]
            for record in chunk:
                if hashlib.sha256((data_root / record["audio"]).read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError(f"Evaluation audio changed: {record['audio']}")
            result = invoke(["transcribe", "--model", str(model_path), "--format", "json",
                             *(str(data_root / record["audio"]) for record in chunk)])
            try:
                decoded = json_stream(result.stdout)
            except json.JSONDecodeError:
                decoded = []
            if result.returncode:
                decoded = []
                for record in chunk:
                    individual = invoke(["transcribe", "--model", str(model_path), "--format", "json", str(data_root / record["audio"])])
                    if individual.returncode:
                        failures[record["audio"]] = individual.stderr.strip()
                        print(f"[!] PureScribe {record['audio']}: {individual.stderr.strip()}", flush=True)
                    else:
                        decoded.extend(json_stream(individual.stdout))
            for item in decoded:
                audio = Path(item["audio"]).relative_to(data_root).as_posix()
                if audio not in {record["audio"] for record in chunk} or audio in predictions:
                    raise ValueError(f"Unexpected or duplicate prediction: {audio}")
                predictions[audio] = item["text"]
                stream.write(json.dumps({**item, "audio": audio}, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"PureScribe evaluation: {min(offset + 32, len(records))}/{len(records)}", flush=True)
    with (output / "reference.tsv").open(encoding="utf-8", newline="") as stream:
        references = dict(csv.reader(stream, delimiter="\t"))
    metrics = score_predictions(references, predictions, vocabulary)
    metrics.update(model=str(model_path), model_vocabulary=sorted(vocabulary),
                   training_selection=selection,
                   decoding_failures=failures, missing_predictions=sorted(references.keys() - predictions.keys()),
                   training_evaluation_audio_overlap=0, training_evaluation_speaker_overlap=0)
    write_json(output / "metrics.json", metrics)
    write_json(output / "commands.json", commands)
    print(json.dumps({key: metrics[key] for key in ("all", "in_vocabulary", "oov_occurrences", "decoding_failures")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extract", "evaluate"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("D:/CommonVoice"))
    parser.add_argument("--reference", default="cv_path_transcription.txt")
    parser.add_argument("--metadata", default="validated.tsv")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--vocabulary-size", type=int, default=8)
    parser.add_argument("--min-clips", type=int, default=12)
    parser.add_argument("--goal-clips", type=int, default=15)
    parser.add_argument("--max-train", type=int, default=240)
    parser.add_argument("--max-evaluation", type=int, default=400)
    args = parser.parse_args()
    if not 1 <= args.min_clips <= args.goal_clips <= CLIPS_PER_WORD:
        parser.error("Require 1 <= min-clips <= goal-clips <= CLIPS_PER_WORD")
    if min(args.max_train, args.max_evaluation, args.vocabulary_size) < 1:
        parser.error("Pilot limits must be positive")
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "extract":
        extract_training(args.output.resolve())
    else:
        train_and_evaluate(args.output.resolve())


if __name__ == "__main__":
    main()
