from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import wave
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from importlib import metadata
from numbers import Real
from pathlib import Path, PureWindowsPath

TARGET_SR = 16000
PADDING_START_SEC = 0.050
PADDING_END_SEC = 0.120
MIN_CLIP_DURATION_SEC = 0.100

LABEL = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*")
RESERVED_NAMES = {"con", "prn", "aux", "nul"}

SETTINGS_FILE = Path(__file__).with_name("commonvoice_purescribe.settings.json")
DEFAULT_DATA_ROOT = Path(__file__).with_name("corpus")
DEFAULT_OUTPUT = Path("corpus_cv")


def _comparison_word(word: str) -> str:
    word = word.casefold().replace("’", "'").replace("‘", "'")
    return re.sub(r"^[\W_]+|[\W_]+$", "", word)


def match_word_indices(reference_words, whisper_words):
    reference_words = [_comparison_word(word) for word in reference_words]
    whisper_words = [_comparison_word(word) for word in whisper_words]
    matcher = SequenceMatcher(None, reference_words, whisper_words, autojunk=False)
    matches = []
    for reference_start, whisper_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            matches.append((reference_start + offset, whisper_start + offset))
    return matches


def valid_word_bounds(start, end, audio_duration: float):
    for value in (start, end):
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(start) and math.isfinite(end)):
        return None
    if not 0 <= start < end <= audio_duration:
        return None
    return start, end


def save_wav(path: Path, audio, sr: int):
    import numpy as np

    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767).astype(np.int16)
    output = None
    writer = None
    created = False
    try:
        output = path.open("xb")
        created = True
        writer = wave.open(output, "wb")
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sr)
        writer.writeframes(samples.tobytes())
        writer.close()
        output.close()
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except BaseException:
                pass
        if output is not None and not output.closed:
            try:
                output.close()
            except BaseException:
                pass
        if created:
            try:
                path.unlink()
            except BaseException:
                pass
        raise


def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")


def _release_rows(rows, header):
    path = header.index("path")
    sentence = header.index("sentence")
    for row in rows:
        if not row:
            yield row
        elif len(row) == len(header):
            yield [row[path], row[sentence]]
        else:
            yield [row[path] if len(row) > path else "", None, None]


def read_reference_map(path, audio_names=None, audit=None):
    references = {}
    seen = set()
    local_names = None
    if audio_names is not None:
        local_names = {name.casefold(): name for name in audio_names}
    if audit is None:
        audit = Counter()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = stream.readline().rstrip("\r\n").split("\t")
        stream.seek(0)
        release = len(header) > 2 and {"path", "sentence"} <= set(header)
        quoting = csv.QUOTE_NONE if release else csv.QUOTE_MINIMAL
        rows = csv.reader(stream, delimiter="\t", strict=True, quoting=quoting)
        if release:
            next(rows)
            rows = _release_rows(rows, header)
        for index, row in enumerate(rows):
            if index == 0 and row == ["path", "sentence"]:
                continue
            audit["records"] += 1
            if not row:
                audit["blank_records"] += 1
                continue
            if local_names is not None and row[0].casefold() not in local_names:
                audit["outside_validated"] += 1
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
            if local_names is not None:
                name = local_names[name.casefold()]
            references[name] = sentence
    return references


def levenshtein(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], ops[i][0] = i, "D"
    for j in range(1, m + 1):
        dp[0][j], ops[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            substitution = dp[i - 1][j - 1] + cost
            deletion = dp[i - 1][j] + 1
            insertion = dp[i][j - 1] + 1
            best = min(substitution, deletion, insertion)
            dp[i][j] = best
            if best == substitution:
                ops[i][j] = "C" if cost == 0 else "S"
            elif best == deletion:
                ops[i][j] = "D"
            else:
                ops[i][j] = "I"

    i, j = n, m
    substitutions = deletions = insertions = 0
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == "C":
            i, j = i - 1, j - 1
        elif op == "S":
            substitutions += 1
            i, j = i - 1, j - 1
        elif op == "D":
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return substitutions + deletions + insertions, substitutions, deletions, insertions


def scoring_tokens(sentence):
    return [_comparison_word(token) for token in sentence.split() if re.search(r"\w", token)]


def score_predictions(references, predictions, vocabulary):
    total = Counter()
    known = Counter()
    oov = Counter()
    details = []
    for audio, sentence in references.items():
        reference = scoring_tokens(sentence)
        hypothesis = scoring_tokens(predictions.get(audio, ""))
        errors, substitutions, deletions, insertions = levenshtein(reference, hypothesis)
        counts = {
            "sentences": 1,
            "reference_words": len(reference),
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "errors": errors,
            "correct_sentences": int(reference == hypothesis),
        }
        in_vocabulary = all(word in vocabulary for word in reference)
        total.update(counts)
        if in_vocabulary:
            known.update(counts)
        oov.update(word for word in reference if word not in vocabulary)
        details.append({
            "audio": audio,
            "reference": reference,
            "hypothesis": hypothesis,
            "in_vocabulary": in_vocabulary,
            **counts,
        })

    def finish(counts):
        keys = ("sentences", "reference_words", "substitutions", "deletions",
                "insertions", "errors", "correct_sentences")
        result = {key: counts[key] for key in keys}
        result["wer_percent"] = None
        if counts["reference_words"]:
            result["wer_percent"] = 100 * counts["errors"] / counts["reference_words"]
        result["sentence_accuracy_percent"] = None
        if counts["sentences"]:
            result["sentence_accuracy_percent"] = 100 * counts["correct_sentences"] / counts["sentences"]
        return result

    return {
        "all": finish(total),
        "in_vocabulary": finish(known),
        "oov_occurrences": sum(oov.values()),
        "oov_words": dict(oov.most_common()),
        "normalization": "casefold, typographic apostrophes -> ASCII, external punctuation only",
        "per_audio": details,
    }


def official_words(sentence):
    pairs = [(token, _comparison_word(token)) for token in sentence.split()]
    return [(token, label) for token, label in pairs if label]


def read_metadata(path, local_names, references):
    metadata = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = {"path", "client_id", "locale"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            name = local_names.get((row["path"] or "").casefold())
            if name in references:
                if name in metadata:
                    raise ValueError(f"{path}: ambiguous metadata for {name}")
                metadata[name] = (row["client_id"] or "", row["locale"] or "")
    return metadata


def eligible_label(label):
    return bool(LABEL.fullmatch(label)) and label not in RESERVED_NAMES


def clear_previous_output(output, inputs):
    if not output.is_dir() or output.is_symlink():
        raise FileExistsError(f"output exists and is not a plain directory: {output}")
    for path in inputs:
        if path == output or output in path.parents:
            raise ValueError(f"output {output} would contain the input {path}; choose another --output")

    entries = list(output.iterdir())
    ledgers = [e for e in entries if e.name in ("clips.jsonl", "extraction.json") and e.is_file()]
    words = [e for e in entries if e.is_dir() and not e.is_symlink()]
    unexpected = [e for e in entries if e not in ledgers and e not in words]
    for word in words:
        for clip in word.iterdir():
            if clip.is_symlink() or not clip.is_file() or clip.suffix.lower() != ".wav":
                unexpected.append(clip)
    if entries and (not ledgers or unexpected):
        examples = [str(e) for e in unexpected[:3]]
        raise FileExistsError(f"{output} is not an earlier extraction output (no clips.jsonl/extraction.json, "
                              f"or other files such as {examples}); choose another --output")

    removed = 0
    for word in words:
        for clip in word.iterdir():
            clip.unlink()
            removed += 1
        word.rmdir()
    for ledger in ledgers:
        ledger.unlink()
    return removed


def existing_clip_counts(output):
    if not output.is_dir() or output.is_symlink():
        raise FileExistsError(f"resume output is not a plain directory: {output}")
    entries = list(output.iterdir())
    ledgers = [entry for entry in entries if entry.name in ("clips.jsonl", "extraction.json")
               and entry.is_file() and not entry.is_symlink()]
    if not any(entry.name == "clips.jsonl" for entry in ledgers):
        raise FileExistsError(f"resume output has no clips.jsonl checkpoint: {output}")
    unexpected = [entry for entry in entries if entry not in ledgers and
                  (not entry.is_dir() or entry.is_symlink())]
    counts = Counter()
    for word_dir in (entry for entry in entries if entry.is_dir() and not entry.is_symlink()):
        for clip in word_dir.iterdir():
            if clip.is_symlink() or not clip.is_file() or clip.suffix.lower() != ".wav":
                unexpected.append(clip)
            else:
                counts[word_dir.name] += 1
    if unexpected:
        raise FileExistsError(f"resume output contains unexpected files such as {unexpected[:3]}")
    return counts


def completed_audio_names(output, references, usable):
    recorded = defaultdict(set)
    with (output / "clips.jsonl").open("r", encoding="utf-8") as ledger:
        for line_number, line in enumerate(ledger, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{output / 'clips.jsonl'}:{line_number}: invalid JSON") from exc
            name, index = row.get("audio"), row.get("word_index")
            if isinstance(name, str) and isinstance(index, int) and not isinstance(index, bool):
                recorded[name].add(index)
    complete = set()
    for name in usable:
        required = {index for index, (_, label) in enumerate(official_words(references[name]))
                    if eligible_label(label)}
        if required and required.issubset(recorded[name]):
            complete.add(name)
    return complete


def extract(args):
    quota = args.clips_per_word
    minimum = args.min_clips_per_word
    if minimum > quota:
        raise ValueError("--min-clips-per-word cannot exceed --clips-per-word")

    data_root = args.data_root.resolve()
    audio_dir = (args.audio_dir or data_root / "dev").resolve()
    reference_path = (args.transcription_file or data_root / "dev.tsv").resolve()
    metadata_path = (args.metadata_file or data_root / f"{audio_dir.name}.tsv").resolve()
    output = args.output.resolve()
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"audio directory not found: {audio_dir}")
    audio_names = {path.name for path in audio_dir.iterdir()
                   if path.is_file() and path.suffix.lower() == ".mp3"}
    if not audio_names:
        raise ValueError(f"no MP3 files in {audio_dir}")

    reference_audit = Counter()
    references = read_reference_map(reference_path, audio_names, reference_audit)
    if args.metadata_file is not None and not metadata_path.is_file():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")
    metadata = None
    if metadata_path.is_file():
        local_names = {name.casefold(): name for name in audio_names}
        metadata = read_metadata(metadata_path, local_names, references)

    usable = []
    rejected = Counter()
    for name, sentence in sorted(references.items()):
        if not official_words(sentence) or any(char in sentence for char in "\r\n\t"):
            rejected["empty_or_multiline_reference"] += 1
        elif metadata is not None and metadata.get(name, ("", ""))[1] not in ("", "en"):
            rejected["non_english_locale"] += 1
        else:
            usable.append(name)

    def speaker(name):
        if metadata is not None and metadata.get(name, ("", ""))[0]:
            return metadata[name][0]
        return None

    available = defaultdict(set)
    for name in usable:
        for _, label in official_words(references[name]):
            if eligible_label(label) and speaker(name):
                available[label].add(speaker(name))

    settings_file = getattr(args, "settings_file", None)
    if settings_file is not None:
        save_settings(settings_file, {
            "data_root": data_root,
            "audio_dir": audio_dir,
            "transcription_file": reference_path,
            "output": output,
        })

    import whisperx

    accepted = Counter()
    saved = defaultdict(list)
    speakers = defaultdict(set)
    attempted = Counter()
    outcomes = defaultdict(Counter)
    audio_status = Counter()
    skipped = Counter()
    skipped_occurrences = Counter()
    model = aligner = align_metadata = device = None
    processed = 0
    started = time.monotonic()
    resume_existing = getattr(args, "resume_existing", False)
    completed = completed_audio_names(output, references, usable) if resume_existing else set()

    if output.exists():
        if resume_existing:
            accepted.update(existing_clip_counts(output))
            print(f"resuming in {output} ({sum(accepted.values())} existing clips)", flush=True)
        else:
            removed = clear_previous_output(output, (data_root, audio_dir, reference_path, metadata_path))
            print(f"overwriting the previous extraction in {output} ({removed} clips removed)", flush=True)
    elif resume_existing:
        raise FileNotFoundError(f"resume output not found: {output}")
    output.mkdir(parents=True, exist_ok=True)

    stopped = "audio_exhausted"
    ledger_mode = "a" if resume_existing else "x"
    with (output / "clips.jsonl").open(ledger_mode, encoding="utf-8") as ledger:
        try:
            for position, name in enumerate(usable, 1):
                if name in completed:
                    skipped["already_complete"] += 1
                    continue
                words = official_words(references[name])
                eligible = {label for _, label in words if eligible_label(label)}
                wanted = {label for label in eligible if accepted[label] < quota}
                if not wanted:
                    skipped["quota_full" if eligible else "no_eligible_word"] += 1
                    skipped_occurrences.update(label for _, label in words if eligible_label(label))
                    continue
                processed += 1
                attempted.update(wanted)

                status = "transcribed"
                detail = None
                recognized = []
                audio = None
                try:
                    audio = whisperx.load_audio(str(audio_dir / name))
                except Exception as exc:
                    status = "load_error"
                    detail = f"{type(exc).__name__}: {exc}"
                if status == "transcribed":
                    try:
                        if model is None:
                            import torch

                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            model = whisperx.load_model(args.whisper_model, device, language="en")
                            aligner, align_metadata = whisperx.load_align_model("en", device)
                        result = model.transcribe(audio, language="en")
                        aligned = whisperx.align(result["segments"], aligner, align_metadata, audio, device,
                                                 interpolate_method="ignore")
                        recognized = [word for word in aligned["word_segments"]
                                      if isinstance(word.get("word"), str) and _comparison_word(word["word"])]
                    except Exception as exc:
                        status = "whisper_error"
                        detail = f"{type(exc).__name__}: {exc}"
                audio_status[status] += 1

                matches = dict(match_word_indices([token for token, _ in words],
                                                  [word["word"] for word in recognized]))
                duration = len(audio) / TARGET_SR if status == "transcribed" else 0.0
                for index, (token, label) in enumerate(words):
                    if not eligible_label(label):
                        continue
                    row = {
                        "audio": name,
                        "client_id": speaker(name),
                        "word_index": index,
                        "official_word": token,
                        "label": label,
                    }
                    if status != "transcribed":
                        row.update(status=status, detail=detail)
                    elif index not in matches:
                        row["status"] = "unmatched"
                    else:
                        word = recognized[matches[index]]
                        bounds = valid_word_bounds(word.get("start"), word.get("end"), duration)
                        if bounds is None:
                            row["status"] = "invalid_timestamp"
                        elif accepted[label] >= quota:
                            row["status"] = "quota_reached"
                        else:
                            start, end = bounds
                            first = int(max(0.0, start - PADDING_START_SEC) * TARGET_SR)
                            last = min(len(audio), int((end + PADDING_END_SEC) * TARGET_SR))
                            if last - first < int(MIN_CLIP_DURATION_SEC * TARGET_SR):
                                row["status"] = "short_clip"
                            else:
                                relative_wav = f"{label}/{Path(name).stem}_w{index:02d}.wav"
                                wav = output / relative_wav
                                if resume_existing and wav.is_file():
                                    row.update(status="already_exists", wav=relative_wav)
                                else:
                                    wav.parent.mkdir(exist_ok=True)
                                    save_wav(wav, audio[first:last], TARGET_SR)
                                    accepted[label] += 1
                                    saved[label].append(wav)
                                    if speaker(name):
                                        speakers[label].add(speaker(name))
                                    row.update(
                                        status="saved",
                                        whisper_start=start,
                                        whisper_end=end,
                                        clip_start=first / TARGET_SR,
                                        clip_end=last / TARGET_SR,
                                        wav=relative_wav,
                                    )
                    outcomes[label][row["status"]] += 1
                    ledger.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                ledger.flush()
                print(f"[{position}/{len(usable)}] {name} {status}: "
                      f"{sum(accepted.values())} clips, {len(accepted)} words", flush=True)
        except KeyboardInterrupt:
            stopped = "interrupted"
            print("interrupted: pruning and writing the summary", flush=True)

        pruned = ({word: count for word, count in sorted(accepted.items()) if count < minimum}
                  if not resume_existing else {})
        for word in pruned:
            for wav in saved[word]:
                wav.unlink()
                entry = {"label": word, "status": "pruned", "wav": wav.relative_to(output).as_posix()}
                ledger.write(json.dumps(entry, ensure_ascii=False) + "\n")
            (output / word).rmdir()
        for directory in output.iterdir():
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()

    vocabulary = sorted(word for word in accepted if word not in pruned)
    per_word = {}
    for word in sorted(outcomes):
        counts = outcomes[word]
        considered = sum(counts.values())
        failed = counts["unmatched"] + counts["load_error"] + counts["whisper_error"]
        per_word[word] = {
            "kept": word in vocabulary,
            "audio_attempted": attempted[word],
            "occurrences_considered": considered,
            "exact_matches": considered - failed,
            "accepted": accepted[word],
            "speakers_available": len(available[word]) if metadata is not None else None,
            "speakers_used": len(speakers[word]) if metadata is not None else None,
            "not_accepted": {k: v for k, v in counts.items() if k != "saved"},
            "occurrences_in_audio_skipped_quota_full": skipped_occurrences[word],
        }

    summary = {
        "phase": "extract",
        "audio_directory": str(audio_dir),
        "local_mp3_files": len(audio_names),
        "reference_file": str(reference_path),
        "reference_audit": dict(reference_audit),
        "local_audio_with_reference": len(references),
        "metadata_file": str(metadata_path) if metadata is not None else None,
        "rejected": dict(rejected),
        "usable_audio": len(usable),
        "order": "sorted file names",
        "whisper": {
            "package": "whisperx",
            "version": package_version("whisperx"),
            "model": args.whisper_model,
            "device": device,
            "alignment": "load_align_model('en'): default wav2vec2 model, applied to Whisper's own segments",
            "interpolate_method": "ignore",
            "call": "load_model(model, device, language='en'); transcribe(audio, language='en'); "
                    "align(own segments, interpolate_method='ignore'); no initial_prompt, no official text",
        },
        "selection": "an audio is transcribed only if its official text contains an eligible word "
                     "whose quota is not full",
        "clips_per_word": quota,
        "min_clips_per_word": minimum,
        "audio_processed": processed,
        "audio_transcribed": audio_status["transcribed"],
        "audio_skipped": dict(skipped),
        "audio_status": dict(audio_status),
        "stopped": stopped,
        "clip_format": {
            "rate": TARGET_SR,
            "channels": 1,
            "sample": "PCM signed 16 bit",
            "padding_start_seconds": PADDING_START_SEC,
            "padding_end_seconds": PADDING_END_SEC,
            "minimum_seconds": MIN_CLIP_DURATION_SEC,
        },
        "clips_extracted": sum(accepted.values()),
        "clips_kept": sum(accepted[word] for word in vocabulary),
        "vocabulary_size": len(vocabulary),
        "vocabulary": vocabulary,
        "pruned_words": pruned,
        "outcomes_all_words": dict(sum(outcomes.values(), Counter())),
        "per_word": per_word,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    summary_path = output / "extraction.json"
    if resume_existing and summary_path.exists():
        summary_path = output / f"extraction_resume_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_json(summary_path, summary)

    shown = ("audio_transcribed", "stopped", "clips_kept", "vocabulary_size")
    print(json.dumps({key: summary[key] for key in shown}, ensure_ascii=False, indent=2))
    if len(vocabulary) < 2:
        raise ValueError(f"fewer than two words with at least {minimum} clips: do not train this corpus; "
                         "repeat the extraction in a new output directory without interrupting it "
                         "or with a lower --min-clips-per-word")
    print(f"extraction done: {output}", flush=True)


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def load_settings(path):
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: unreadable settings ({exc}); fix or delete the file") from exc
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: settings must be a JSON object; fix or delete the file")
    return settings


def save_settings(path, values):
    text = json.dumps({key: str(value) for key, value in values.items()}, ensure_ascii=False, indent=2)
    Path(path).write_text(text + "\n", encoding="utf-8")


def apply_settings(args, settings):
    explicit_root = args.data_root is not None
    args.data_root = args.data_root or Path(settings.get("data_root", DEFAULT_DATA_ROOT))
    for key, derived in (("audio_dir", "dev"), ("transcription_file", "dev.tsv")):
        if getattr(args, key) is None:
            remembered = None if explicit_root else settings.get(key)
            setattr(args, key, Path(remembered) if remembered else args.data_root / derived)
    if args.output is None:
        args.output = Path(settings.get("output", DEFAULT_OUTPUT))


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the PureScribe training corpus from Common Voice.")
    phases = parser.add_subparsers(dest="phase", required=True)

    e = phases.add_parser("extract", help="WhisperX word timestamps -> labelled WAV training clips")
    e.add_argument("--output", type=Path, default=None,
                   help="corpus directory <output>/<word>/*.wav; an earlier extraction there is overwritten "
                        "(default: remembered, else corpus_cv)")
    e.add_argument("--data-root", type=Path, default=None,
                   help="Common Voice root (default: remembered, else the repository corpus folder)")
    e.add_argument("--audio-dir", type=Path, default=None,
                   help="MP3 directory (default: remembered, else <data-root>/dev)")
    e.add_argument("--transcription-file", type=Path, default=None,
                   help="official Common Voice TSV with path and sentence columns, or <file.mp3>\\t<sentence> "
                        "(default: remembered, else <data-root>/dev.tsv)")
    e.add_argument("--metadata-file", type=Path, default=None,
                   help="optional TSV with path, client_id, locale (default: <data-root>/<audio-dir name>.tsv)")
    e.add_argument("--clips-per-word", type=positive, default=23, help="maximum clips per word (default: 23)")
    e.add_argument("--min-clips-per-word", type=positive, default=12,
                   help="words with fewer clips are pruned at the end (default: 12)")
    e.add_argument("--whisper-model", default="base", help="WhisperX (faster-whisper) model name (default: base)")
    e.add_argument("--resume-existing", action="store_true",
                   help="append to an existing extraction, preserving its WAVs and per-word quotas")
    e.set_defaults(func=extract)

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        apply_settings(args, load_settings(SETTINGS_FILE))
        args.settings_file = SETTINGS_FILE
        args.func(args)
    except (FileExistsError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
