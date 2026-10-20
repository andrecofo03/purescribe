"""Offline construction of the PureScribe training corpus from Common Voice.

  extract  autonomous Whisper word timestamps -> exact matches -> WAV clips labelled
           with the official Common Voice word, at most --clips-per-word per word;
           words left with fewer than --min-clips-per-word clips are pruned
  then     purescribe train <output> -o <model>.psm   (existing CLI)

Every Common Voice clip is training data: the system is evaluated downstream on a
different dataset (DeepSpeak), so no Common Voice train/validation split is made.
Whisper is imported only by `extract` and never receives the official text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
import wave
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from numbers import Real
from pathlib import Path, PureWindowsPath

TARGET_SR = 16000
PADDING_START_SEC = 0.050
PADDING_END_SEC = 0.120
MIN_CLIP_DURATION_SEC = 0.100

# A class label must also be a safe Windows directory name: letters, with
# apostrophes or hyphens only between letters ("we're", "re-sign").
LABEL = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*")
RESERVED_NAMES = {"con", "prn", "aux", "nul"}

# Paths given once on the command line are remembered here for the next runs.
SETTINGS_FILE = Path(__file__).with_name("commonvoice_purescribe.settings.json")
DEFAULT_DATA_ROOT = Path(__file__).with_name("corpus")
DEFAULT_OUTPUT = Path("corpus_cv")


# ---------------------------------------------------------------- words, timestamps, clips

def _comparison_word(word: str) -> str:
    word = word.casefold().replace("’", "'").replace("‘", "'")
    return re.sub(r"^[\W_]+|[\W_]+$", "", word)


def match_word_indices(reference_words, whisper_words):
    """Exact, monotonic matches of normalized words; a Whisper word is never reused."""
    reference_words = [_comparison_word(word) for word in reference_words]
    whisper_words = [_comparison_word(word) for word in whisper_words]
    matches = []
    for reference_start, whisper_start, size in SequenceMatcher(
        None, reference_words, whisper_words, autojunk=False
    ).get_matching_blocks():
        matches.extend(
            (reference_start + offset, whisper_start + offset)
            for offset in range(size)
        )
    return matches


def valid_word_bounds(start, end, audio_duration: float):
    if (
        isinstance(start, bool)
        or not isinstance(start, Real)
        or isinstance(end, bool)
        or not isinstance(end, Real)
    ):
        return None
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or not 0 <= start < end <= audio_duration
    ):
        return None
    return start, end


def save_wav(path: Path, audio, sr: int):
    """Write mono PCM16 exclusively; on failure remove only the file created here."""
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


# ---------------------------------------------------------------- sources, scoring

def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)  # fails before creating the file
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")


def _release_rows(rows, header):
    path, sentence = header.index("path"), header.index("sentence")
    for row in rows:
        if not row or len(row) == len(header):
            yield [row[path], row[sentence]] if row else row
        else:
            # Wrong field count: three fields, so the record is rejected if its audio is selected.
            yield [row[path] if len(row) > path else "", None, None]


def read_reference_map(path, audio_names=None, audit=None):
    references = {}
    seen = set()
    local_names = {name.casefold(): name for name in audio_names} if audio_names is not None else None
    if audit is None:
        audit = Counter()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = stream.readline().rstrip("\r\n").split("\t")
        stream.seek(0)
        release = len(header) > 2 and {"path", "sentence"} <= set(header)
        rows = csv.reader(stream, delimiter="\t", strict=True,
                          quoting=csv.QUOTE_NONE if release else csv.QUOTE_MINIMAL)
        if release:
            # Common Voice release TSV (dev.tsv): unquoted fields, path and sentence by column name.
            next(rows)
            rows = _release_rows(rows, header)
        for index, row in enumerate(rows):
            if index == 0 and row == ["path", "sentence"]:
                continue
            audit["records"] += 1
            if not row:
                audit["blank_records"] += 1
                continue
            # Rows of audio outside the audio directory may have extra export columns; they never enter a run.
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
            references[local_names[name.casefold()] if local_names is not None else name] = sentence
    return references


def levenshtein(ref, hyp):
    """Word edit distance with its substitutions, deletions and insertions."""
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
    """Comparison forms of the tokens that contain a word character (scoring as in the validated pilot)."""
    return [_comparison_word(token) for token in sentence.split() if re.search(r"\w", token)]


def score_predictions(references, predictions, vocabulary):
    """WER, S/D/I and exact sentences, for all audio and for fully in-vocabulary sentences."""
    total, known = Counter(), Counter()
    oov, details = Counter(), []
    for audio, sentence in references.items():
        reference = scoring_tokens(sentence)
        hypothesis = scoring_tokens(predictions.get(audio, ""))
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


# ---------------------------------------------------------------- extraction

def official_words(sentence):
    """(token, label) pairs of an official sentence; position = word index."""
    pairs = [(token, _comparison_word(token)) for token in sentence.split()]
    return [(token, label) for token, label in pairs if label]


def read_metadata(path, local_names, references):
    """client_id and locale of the referenced local audio; nothing else is used."""
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
    """Empty an earlier extraction output; anything else is refused, so no unrelated data is deleted."""
    if not output.is_dir() or output.is_symlink():
        raise FileExistsError(f"output exists and is not a plain directory: {output}")
    for path in inputs:
        if path == output or output in path.parents:
            raise ValueError(f"output {output} would contain the input {path}; choose another --output")
    entries = list(output.iterdir())
    ledgers = [e for e in entries if e.name in ("clips.jsonl", "extraction.json") and e.is_file()]
    words = [e for e in entries if e.is_dir() and not e.is_symlink()]
    unexpected = [e for e in entries if e not in ledgers and e not in words]
    unexpected += [clip for word in words for clip in word.iterdir()
                   if clip.is_symlink() or not clip.is_file() or clip.suffix.lower() != ".wav"]
    if entries and (not ledgers or unexpected):
        raise FileExistsError(f"{output} is not an earlier extraction output (no clips.jsonl/extraction.json, "
                              f"or other files such as {[str(e) for e in unexpected[:3]]}); "
                              "choose another --output")
    removed = 0
    for word in words:
        for clip in word.iterdir():
            clip.unlink()
            removed += 1
        word.rmdir()
    for ledger in ledgers:
        ledger.unlink()
    return removed


def extract(args):
    quota, minimum = args.clips_per_word, args.min_clips_per_word
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
    # Metadata only adds client_id (speaker statistics) and locale; only the default file is optional.
    if args.metadata_file is not None and not metadata_path.is_file():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")
    metadata = None
    if metadata_path.is_file():
        local_names = {name.casefold(): name for name in audio_names}
        metadata = read_metadata(metadata_path, local_names, references)
    usable, rejected = [], Counter()
    for name, sentence in sorted(references.items()):
        if not official_words(sentence) or any(char in sentence for char in "\r\n\t"):
            rejected["empty_or_multiline_reference"] += 1
        elif metadata is not None and metadata.get(name, ("", ""))[1] not in ("", "en"):
            rejected["non_english_locale"] += 1
        else:
            usable.append(name)
    # Seeded order spreads consecutive recordings of one speaker over the run.
    random.Random(args.seed).shuffle(usable)

    def speaker(name):
        return metadata[name][0] if metadata is not None and metadata.get(name, ("", ""))[0] else None

    available = defaultdict(set)  # known speakers whose official text contains the word
    for name in usable:
        for _, label in official_words(references[name]):
            if eligible_label(label) and speaker(name):
                available[label].add(speaker(name))

    settings_file = getattr(args, "settings_file", None)
    if settings_file is not None:
        save_settings(settings_file, {"data_root": data_root, "audio_dir": audio_dir,
                                      "transcription_file": reference_path, "output": output})

    import whisper

    accepted, saved, speakers = Counter(), defaultdict(list), defaultdict(set)
    attempted, outcomes, audio_status = Counter(), defaultdict(Counter), Counter()
    skipped_occurrences = Counter()  # occurrences in audio skipped because every quota was full
    model, processed, skipped, started = None, 0, Counter(), time.monotonic()
    if output.exists():
        removed = clear_previous_output(output, (data_root, audio_dir, reference_path, metadata_path))
        print(f"overwriting the previous extraction in {output} ({removed} clips removed)", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    stopped = "audio_exhausted"
    with (output / "clips.jsonl").open("x", encoding="utf-8") as ledger:
        try:
            for position, name in enumerate(usable, 1):
                words = official_words(references[name])
                # The official text is used only to skip audio that cannot add a clip below quota.
                eligible = {label for _, label in words if eligible_label(label)}
                wanted = {label for label in eligible if accepted[label] < quota}
                if not wanted:
                    skipped["quota_full" if eligible else "no_eligible_word"] += 1
                    skipped_occurrences.update(label for _, label in words if eligible_label(label))
                    continue
                processed += 1
                attempted.update(wanted)
                status, detail, recognized, audio = "transcribed", None, [], None
                try:
                    audio = whisper.load_audio(str(audio_dir / name))
                except Exception as exc:
                    status, detail = "load_error", f"{type(exc).__name__}: {exc}"
                if status == "transcribed":
                    try:
                        if model is None:
                            model = whisper.load_model(args.whisper_model)
                        # Autonomous transcription: audio only, no prompt, no expected words.
                        result = model.transcribe(audio, language="en", word_timestamps=True,
                                                  fp16=model.device.type == "cuda", verbose=None)
                        recognized = [word for segment in result["segments"] for word in segment.get("words", [])
                                      if isinstance(word.get("word"), str) and _comparison_word(word["word"])]
                    except Exception as exc:
                        status, detail = "whisper_error", f"{type(exc).__name__}: {exc}"
                audio_status[status] += 1
                matches = dict(match_word_indices([token for token, _ in words],
                                                  [word["word"] for word in recognized]))
                duration = len(audio) / TARGET_SR if status == "transcribed" else 0.0
                for index, (token, label) in enumerate(words):
                    if not eligible_label(label):
                        continue
                    row = {"audio": name, "client_id": speaker(name), "word_index": index,
                           "official_word": token, "label": label}
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
                                wav = output / label / f"{Path(name).stem}_w{index:02d}.wav"
                                wav.parent.mkdir(exist_ok=True)
                                save_wav(wav, audio[first:last], TARGET_SR)
                                accepted[label] += 1
                                saved[label].append(wav)
                                if speaker(name):
                                    speakers[label].add(speaker(name))
                                row.update(status="saved", whisper_start=start, whisper_end=end,
                                           clip_start=first / TARGET_SR, clip_end=last / TARGET_SR,
                                           wav=wav.relative_to(output).as_posix())
                    outcomes[label][row["status"]] += 1
                    ledger.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                ledger.flush()
                print(f"[{position}/{len(usable)}] {name} {status}: "
                      f"{sum(accepted.values())} clips, {len(accepted)} words", flush=True)
        except KeyboardInterrupt:
            # Ctrl+C ends the run early; the clips saved so far are still pruned and summarised.
            stopped = "interrupted"
            print("interrupted: pruning and writing the summary", flush=True)

        # Too few clips make the diagonal covariances collapse: remove only the files saved above.
        pruned = {word: count for word, count in sorted(accepted.items()) if count < minimum}
        for word in pruned:
            for wav in saved[word]:
                wav.unlink()
                ledger.write(json.dumps({"label": word, "status": "pruned",
                                         "wav": wav.relative_to(output).as_posix()},
                                        ensure_ascii=False) + "\n")
            (output / word).rmdir()
        for directory in output.iterdir():  # a save interrupted before its first clip
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()

    vocabulary = sorted(word for word in accepted if word not in pruned)
    per_word = {}  # every word met in transcribed audio, kept, pruned or never saved
    for word in sorted(outcomes):
        counts = outcomes[word]
        considered = sum(counts.values())
        failed = counts["unmatched"] + counts["load_error"] + counts["whisper_error"]
        per_word[word] = {"kept": word in vocabulary, "audio_attempted": attempted[word],
                          "occurrences_considered": considered, "exact_matches": considered - failed,
                          "accepted": accepted[word],
                          "speakers_available": len(available[word]) if metadata is not None else None,
                          "speakers_used": len(speakers[word]) if metadata is not None else None,
                          "not_accepted": {k: v for k, v in counts.items() if k != "saved"},
                          "occurrences_in_audio_skipped_quota_full": skipped_occurrences[word]}
    summary = {
        "phase": "extract",
        "audio_directory": str(audio_dir), "local_mp3_files": len(audio_names),
        "reference_file": str(reference_path), "reference_audit": dict(reference_audit),
        "local_audio_with_reference": len(references),
        "metadata_file": str(metadata_path) if metadata is not None else None,
        "rejected": dict(rejected), "usable_audio": len(usable),
        "order": {"rule": "sorted file names, then random.Random(seed).shuffle", "seed": args.seed},
        "whisper": {"package": "openai-whisper", "version": getattr(whisper, "__version__", None),
                    "model": args.whisper_model,
                    "device": str(model.device) if model is not None else None,
                    "call": "transcribe(audio, language='en', word_timestamps=True); no initial_prompt, no official text"},
        "selection": "an audio is transcribed only if its official text contains an eligible word "
                     "whose quota is not full",
        "clips_per_word": quota, "min_clips_per_word": minimum,
        "audio_processed": processed, "audio_transcribed": audio_status["transcribed"],
        "audio_skipped": dict(skipped),
        "audio_status": dict(audio_status),
        "stopped": stopped,
        "clip_format": {"rate": TARGET_SR, "channels": 1, "sample": "PCM signed 16 bit",
                        "padding_start_seconds": PADDING_START_SEC, "padding_end_seconds": PADDING_END_SEC,
                        "minimum_seconds": MIN_CLIP_DURATION_SEC},
        "clips_extracted": sum(accepted.values()),
        "clips_kept": sum(accepted[word] for word in vocabulary),
        "vocabulary_size": len(vocabulary), "vocabulary": vocabulary,
        "pruned_words": pruned,
        "outcomes_all_words": dict(sum(outcomes.values(), Counter())),
        "per_word": per_word,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    write_json(output / "extraction.json", summary)
    print(json.dumps({key: summary[key] for key in ("audio_transcribed", "stopped", "clips_kept",
                                                    "vocabulary_size")}, ensure_ascii=False, indent=2))
    if len(vocabulary) < 2:
        raise ValueError(f"fewer than two words with at least {minimum} clips: do not train this corpus; "
                         "repeat the extraction in a new output directory without interrupting it "
                         "or with a lower --min-clips-per-word")
    print(f"extraction done: {output}", flush=True)


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
    Path(path).write_text(json.dumps({key: str(value) for key, value in values.items()},
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_settings(args, settings):
    """Command line > paths derived from an explicit --data-root > remembered > defaults."""
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)

    e = phases.add_parser("extract", help="Whisper word timestamps -> labelled WAV training clips")
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
    e.add_argument("--seed", type=int, default=20261014, help="seed of the audio processing order")
    e.add_argument("--whisper-model", default="base", help="Whisper model name (default: base)")
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
