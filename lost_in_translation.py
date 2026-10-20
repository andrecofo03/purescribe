"""Final thesis pipeline: "Lost in Translation" with PureScribe as the audio branch.

  phase 5  transcribe  PureScribe transcribes the audio track of each baseline video
                       (ffmpeg extraction inside purescribe.audio.load_audio)
  phase 6  compare     PureScribe and the baseline's Whisper transcripts against the
                       baseline's lip-reading transcripts, with the baseline protocol:
                       standardize_text, Levenshtein / max length, threshold

No ASR model other than PureScribe runs here. The baseline transcripts are the text files
written by the baseline run (transcribe.py --save-txt):
  <transcripts>/whisper/<subset>/<real|fake>/<video>.txt   audio branch (Whisper)
  <transcripts>/avsr/<subset>/<real|fake>/<video>.txt      video branch (Auto-AVSR)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import statistics
import string
import sys
import time
from collections import Counter
from pathlib import Path

LABELS = ("real", "fake")  # baseline ground truth = folder name; "fake" is the positive class
METHODS = ("diff2lip", "latentsync", "facefusion", "hellomeme", "liveportrait", "memo")
TSV_COLUMNS = ["video", "status", "purescribe_transcription", "detail", "model_sha256"]
TSV_HEADER = "\t".join(TSV_COLUMNS) + "\n"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)  # fails before creating the file
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")


def baseline_videos(transcripts, subset):
    """{"<gt>/<name>.mp4": (whisper_txt, avsr_txt)} for videos the baseline transcribed in both branches."""
    videos, incomplete = {}, []
    for label in LABELS:
        whisper_dir = Path(transcripts) / "whisper" / subset / label
        avsr_dir = Path(transcripts) / "avsr" / subset / label
        whisper = {p.stem: p for p in whisper_dir.glob("*.txt")} if whisper_dir.is_dir() else {}
        avsr = {p.stem: p for p in avsr_dir.glob("*.txt")} if avsr_dir.is_dir() else {}
        incomplete += [f"{label}/{stem}" for stem in sorted(whisper.keys() ^ avsr.keys())]
        for stem in sorted(whisper.keys() & avsr.keys()):
            videos[f"{label}/{stem}.mp4"] = (whisper[stem], avsr[stem])
    if not videos:
        raise ValueError(f"no baseline transcripts under {transcripts} for subset {subset!r}")
    return videos, incomplete


def completed_rows(output, videos, videos_dir, model_hash):
    """Rows of an earlier phase 5 run on the same inputs, after cutting off an interrupted last row.

    Every complete row must be the next baseline video, in order, with this model's hash; anything
    else stops the run with the file untouched. Only an unterminated tail (a row whose write was
    interrupted) is removed, and its video is transcribed again.
    """
    data = output.read_bytes()
    text = data.decode("utf-8", errors="surrogateescape")
    lines = io.StringIO(text, newline="")
    consumed, ends, rows = 0, [], []

    def feed():
        nonlocal consumed
        for line in lines:
            consumed += len(line)
            yield line

    reader = csv.reader(feed(), delimiter="\t", strict=True)
    while True:
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error:  # unterminated quoted field: the write stopped inside it
            break
        if not text[:consumed].endswith("\n"):
            break  # the row terminator was never written
        rows.append(row)
        ends.append(consumed)
    complete = ends[-1] if ends else 0

    if not rows:
        if TSV_HEADER.startswith(text):  # empty, or a header whose write was cut off
            return [], 0, len(data)
        raise ValueError(f"{output}: not a phase 5 output (unreadable header)")
    if rows[0] != TSV_COLUMNS:
        raise ValueError(f"{output}: not a phase 5 output (header {rows[0]})")
    done = rows[1:]
    if len(done) > len(videos):
        raise ValueError(f"{output}: {len(done)} rows for {len(videos)} baseline videos")
    for index, (row, video) in enumerate(zip(done, videos), 2):
        if len(row) != len(TSV_COLUMNS):
            raise ValueError(f"{output}: line {index} has {len(row)} fields")
        if row[0] != video:
            raise ValueError(f"{output}: line {index} is {row[0]}, expected {video} (different order or videos)")
        if row[4] != model_hash:
            raise ValueError(f"{output}: line {index} comes from another model ({row[4]}, now {model_hash})")
        path = videos_dir / video
        expected = {"ok": str(path), "missing": f"not found: {path}"}
        if row[1] not in ("ok", "missing", "error") or row[3] != expected.get(row[1], row[3]):
            raise ValueError(f"{output}: line {index} ({row[1]}, {row[3]}) does not match {videos_dir}")
    complete_bytes = len(text[:complete].encode("utf-8", errors="surrogateescape"))
    return done, complete_bytes, len(data) - complete_bytes


def transcribe(args):
    from purescribe.audio import load_audio
    from purescribe.recognizer import Recognizer

    output, model_path, videos_dir = args.output.resolve(), args.model.resolve(), args.videos_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"output already exists: {output} (use --resume to continue it)")
    if not videos_dir.is_dir():
        raise NotADirectoryError(f"video directory not found: {videos_dir}")
    # Only the names of the baseline transcript files are used here, never their content.
    videos = list(baseline_videos(args.baseline_transcripts, args.subset)[0])
    model_hash = sha256(model_path)
    done, keep_bytes, cut_bytes = [], 0, 0
    if output.exists():
        done, keep_bytes, cut_bytes = completed_rows(output, videos, videos_dir, model_hash)
    print(f"loading model {model_path}", flush=True)
    started = time.monotonic()
    recognizer = Recognizer.load(model_path)
    print(f"model loaded in {time.monotonic() - started:.1f} s", flush=True)

    if output.exists():
        if cut_bytes:
            with output.open("r+b") as stream:
                stream.truncate(keep_bytes)
            print(f"removed an incomplete last row ({cut_bytes} bytes) from {output}", flush=True)
        print(f"resuming {output}: {len(done)} of {len(videos)} videos already complete", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(row[1] for row in done)
    with output.open("a" if output.exists() else "x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        if keep_bytes == 0:
            writer.writerow(TSV_COLUMNS)
            stream.flush()
        for position, video in enumerate(videos[len(done):], len(done) + 1):
            tag = f"[{position}/{len(videos)}] {video}"
            path = videos_dir / video
            started = time.monotonic()
            if not path.is_file():
                status, text, detail = "missing", "", f"not found: {path}"
            else:
                try:
                    print(f"{tag}: extracting audio", flush=True)
                    wave = load_audio(path, rate=recognizer.feature_config.rate)
                    print(f"{tag}: {wave.duration:.2f} s of audio, decoding", flush=True)
                    status, text, detail = "ok", recognizer.transcribe(wave).text, str(path)
                except Exception as exc:
                    status, text, detail = "error", "", f"{type(exc).__name__}: {exc}"
            writer.writerow([video, status, text, detail, model_hash])
            stream.flush()
            counts[status] += 1
            print(f"{tag}: {status} in {time.monotonic() - started:.1f} s: {text if status == 'ok' else detail}",
                  flush=True)
    print(f"phase 5 done: {output} {dict(counts)}", flush=True)


def run_transcribe(args):
    try:
        transcribe(args)
    except KeyboardInterrupt:
        # Rows already written are complete; the video in progress has no row.
        print(f"interrupted: rerun the same command with --resume to continue {args.output}", flush=True)
        raise


# The three functions below reproduce the baseline protocol (DeepSpeak text_utils.py, 2026-09-20).
def standardize_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    text = re.sub(r"\b(um|uh|hmm|mm|mhm|mmm)\b", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(first, second):
    """Levenshtein distance between two strings (unit costs)."""
    previous = list(range(len(second) + 1))
    for i, char in enumerate(first, 1):
        current = [i]
        for j, other in enumerate(second, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def normalized_levenshtein(first, second):
    first = str(first).strip() if first and str(first).lower() != "nan" else ""
    second = str(second).strip() if second and str(second).lower() != "nan" else ""
    if not first and not second:
        return 0.0
    if not first or not second:
        return 1.0
    return round(edit_distance(first, second) / max(len(first), len(second)), 4)


def method_of(name, ground_truth):
    if ground_truth == "real":
        return "authentic"
    return next((prefix for prefix in METHODS if name.lower().startswith(prefix)), "other_fake")


def read_purescribe(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != TSV_COLUMNS:
            raise ValueError(f"{path}: not a phase 5 output")
        rows = {}
        for row in reader:
            if row["video"] in rows:
                raise ValueError(f"{path}: duplicate video {row['video']}")
            rows[row["video"]] = row
    models = {row["model_sha256"] for row in rows.values()}
    if len(models) > 1:
        raise ValueError(f"{path}: transcripts come from different models")
    return rows, models.pop() if models else None


def distance_stats(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.fmean(values), "max": max(values)}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate(rows, system):
    predicted = [(row["ground_truth"], row[system]["prediction"]) for row in rows]
    tp = sum(truth == "fake" and guess == "fake" for truth, guess in predicted)
    tn = sum(truth == "real" and guess == "real" for truth, guess in predicted)
    fp = sum(truth == "real" and guess == "fake" for truth, guess in predicted)
    fn = sum(truth == "fake" and guess == "real" for truth, guess in predicted)
    by_method = {}
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        correct = sum(row[system]["prediction"] == row["ground_truth"] for row in group)
        by_method[method] = {"n": len(group), "correct": correct, "accuracy": ratio(correct, len(group)),
                             "distance": distance_stats([row[system]["distance"] for row in group])}
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": ratio(tp + tn, len(rows)), "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "distance_by_ground_truth": {label: distance_stats([row[system]["distance"] for row in rows
                                                                if row["ground_truth"] == label])
                                         for label in LABELS},
            "by_method": by_method}


def avsr_export(args, videos):
    """Manifest and {video: entry} of a new video-only Auto-AVSR export (avsr_export.py in DeepSpeak)."""
    if args.avsr_manifest is None or args.model is None:
        raise ValueError("--avsr-transcripts needs --avsr-manifest and --model")
    manifest = json.loads(Path(args.avsr_manifest).read_text(encoding="utf-8"))
    if manifest["parameters"]["subset"] != args.subset:
        raise ValueError(f"{args.avsr_manifest}: subset {manifest['parameters']['subset']!r}, not {args.subset!r}")
    entries = {}
    for entry in manifest["videos"]:
        if entry["key"] in entries or entry["key"] not in videos:
            raise ValueError(f"{args.avsr_manifest}: duplicate or foreign video {entry['key']}")
        entries[entry["key"]] = entry
    return manifest, entries


def compare(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if args.avsr_manifest is not None and args.avsr_transcripts is None:
        raise ValueError("--avsr-manifest needs --avsr-transcripts")
    videos, incomplete = baseline_videos(args.baseline_transcripts, args.subset)
    purescribe, model_hash = read_purescribe(args.purescribe)
    unknown = sorted(purescribe.keys() - videos.keys())
    if unknown:
        raise ValueError(f"{args.purescribe}: videos without baseline transcripts: {unknown[:5]}")
    if args.model is not None and sha256(args.model) != model_hash:
        raise ValueError(f"{args.purescribe}: transcripts come from model {model_hash}, not {args.model}")
    manifest, avsr = avsr_export(args, videos) if args.avsr_transcripts is not None else (None, None)

    rows, excluded, digest = [], Counter(), hashlib.sha256()
    excluded_avsr, excluded_avsr_videos = Counter(), []
    for video, (whisper_txt, avsr_txt) in videos.items():
        status = purescribe[video]["status"] if video in purescribe else "not_transcribed"
        if status != "ok":
            excluded[status] += 1  # excluded for both systems: the comparison keeps the same videos
            continue
        label, name = video.split("/", 1)
        if avsr is None:
            video_text = avsr_txt.read_text(encoding="utf-8")
        else:
            entry = avsr.get(video)
            if entry is None:
                raise ValueError(f"{args.avsr_manifest}: no entry for {video}")
            if entry["status"] != "ok":  # not a lip-reading text: excluded for both systems
                excluded_avsr[entry["status"]] += 1
                excluded_avsr_videos.append({"video": video, "status": entry["status"], "reason": entry.get("reason")})
                continue
            data = (Path(args.avsr_transcripts) / "avsr" / args.subset / label / f"{name[:-4]}.txt").read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["text_sha256"]:
                raise ValueError(f"{args.avsr_manifest}: text of {video} does not match its sha256")
            video_text = data.decode("utf-8")
        whisper_text = whisper_txt.read_text(encoding="utf-8")
        digest.update(f"{video}\0{whisper_text}\0{video_text}\0".encode("utf-8"))
        row = {"video": video, "ground_truth": label, "method": method_of(name, label),
               "video_transcription": video_text}
        # The baseline files hold the standardized texts it compared; PureScribe's raw output is
        # standardized here exactly as the baseline standardized Whisper's raw output.
        for system, audio in (("whisper", whisper_text),
                              ("purescribe", standardize_text(purescribe[video]["purescribe_transcription"]))):
            distance = normalized_levenshtein(audio, video_text)
            row[system] = {"audio_transcription": audio, "distance": distance,
                           "prediction": "fake" if distance > args.threshold else "real"}
        rows.append(row)
    if not rows:
        raise ValueError("no video has both baseline transcripts and a PureScribe transcription")

    whisper, pure = evaluate(rows, "whisper"), evaluate(rows, "purescribe")
    change = {}
    for key in ("accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"):
        if whisper[key] is None or pure[key] is None:
            change[key] = {"absolute": None, "relative": None}
        else:
            change[key] = {"absolute": pure[key] - whisper[key],
                           "relative": ratio(pure[key] - whisper[key], whisper[key])}
    result = {
        "phase": "6 compare",
        "inputs": {"baseline_transcripts": str(Path(args.baseline_transcripts).resolve()), "subset": args.subset,
                   "baseline_texts_sha256": digest.hexdigest(),
                   "purescribe_tsv": str(args.purescribe.resolve()), "purescribe_sha256": sha256(args.purescribe),
                   "purescribe_model_sha256": model_hash},
        "protocol": {"video_branch": "baseline lip-reading transcripts, unchanged",
                     "audio_branch_baseline": "baseline Whisper transcripts, unchanged",
                     "normalization": "baseline standardize_text: lower case, [..] and (..) removed, um/uh/hmm/mm/mhm/mmm "
                                      "removed, ASCII punctuation removed, whitespace collapsed",
                     "distance": "baseline normalized_levenshtein: Levenshtein / max length on characters, "
                                 "1.0 if only one text is empty, rounded to 4 decimals",
                     "threshold": args.threshold, "decision": "fake if distance > threshold",
                     "positive_class": "fake"},
        "videos": {"baseline_videos": len(videos), "evaluated": len(rows), "excluded": dict(excluded),
                   "baseline_incomplete": incomplete,
                   "empty_video_transcriptions": sum(not row["video_transcription"].strip() for row in rows),
                   "ground_truth": dict(Counter(row["ground_truth"] for row in rows)),
                   "method": dict(Counter(row["method"] for row in rows))},
        "whisper_baseline": whisper,
        "purescribe": pure,
        "purescribe_minus_whisper": change,
        "per_video": rows,
    }
    if avsr is not None:  # new keys only in this mode: the historical JSON stays as it was
        result["inputs"].update({
            "avsr_transcripts": str(Path(args.avsr_transcripts).resolve()),
            "avsr_manifest": str(Path(args.avsr_manifest).resolve()), "avsr_manifest_sha256": sha256(args.avsr_manifest),
            "avsr_checkpoint": manifest["parameters"]["checkpoint"],
            "avsr_preprocessing": manifest["parameters"]["preprocessing"],
            "purescribe_model": str(Path(args.model).resolve())})
        result["protocol"]["video_branch"] = ("new video-only Auto-AVSR export (avsr_export.py), shared by Whisper "
                                              "and PureScribe; a separate experiment from the baseline texts")
        result["videos"]["excluded_avsr"] = dict(excluded_avsr)
        result["videos"]["excluded_avsr_videos"] = excluded_avsr_videos
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    for system, metrics in (("Whisper", whisper), ("PureScribe", pure)):
        print(f"{system:<10} acc={metrics['accuracy']} prec={metrics['precision']} rec={metrics['recall']} "
              f"f1={metrics['f1']} TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"phase 6 done: {output} ({len(rows)} videos, excluded {dict(excluded)})", flush=True)


def threshold(value):
    number = float(value)
    if not 0.0 <= number <= 1.0:  # also rejects NaN
        raise argparse.ArgumentTypeError("must be a number between 0 and 1")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)

    t = phases.add_parser("transcribe", help="phase 5: PureScribe transcription of the videos' audio")
    t.add_argument("--model", type=Path, required=True, help="PureScribe model (.psm) from phase 3")
    t.add_argument("--baseline-transcripts", type=Path, required=True,
                   help="baseline 'transcriptions' directory; only file names are read")
    t.add_argument("--subset", default="test")
    t.add_argument("--videos-dir", type=Path, required=True, help="directory with real/ and fake/ videos")
    t.add_argument("--output", type=Path, required=True,
                   help="new TSV file (an existing one is refused unless --resume)")
    t.add_argument("--resume", action="store_true",
                   help="continue an existing TSV of the same videos and model: complete rows are kept "
                        "and skipped, an interrupted last row is removed and redone")
    t.set_defaults(func=run_transcribe)

    c = phases.add_parser("compare", help="phase 6: compare Whisper and PureScribe against lip reading")
    c.add_argument("--baseline-transcripts", type=Path, required=True)
    c.add_argument("--subset", default="test")
    c.add_argument("--purescribe", type=Path, required=True, help="TSV written by phase 5")
    c.add_argument("--threshold", type=threshold, default=0.5)
    c.add_argument("--output", type=Path, required=True, help="new JSON file")
    c.add_argument("--avsr-transcripts", type=Path,
                   help="output root of a new video-only Auto-AVSR export, instead of the baseline lip-reading texts")
    c.add_argument("--avsr-manifest", type=Path, help="manifest.json of that export (required with --avsr-transcripts)")
    c.add_argument("--model", type=Path,
                   help="PureScribe model whose SHA-256 must match the TSV (required with --avsr-transcripts)")
    c.set_defaults(func=compare)

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        args.func(args)
    except (FileExistsError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
