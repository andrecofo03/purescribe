from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import string
import sys
import time
from collections import Counter
from pathlib import Path

LABELS = ("real", "fake")
METHODS = ("diff2lip", "latentsync", "facefusion", "hellomeme", "liveportrait", "memo")
TSV_COLUMNS = ["video", "status", "purescribe_transcription", "detail", "model_sha256"]
OLD_TSV_COLUMNS = TSV_COLUMNS
NO_HASH_TSV_COLUMNS = TSV_COLUMNS[:4]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")


def baseline_videos(transcripts, subset):
    videos = {}
    incomplete = []
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


def transcribe(args):
    from purescribe.audio import load_audio
    from purescribe.recognizer import Recognizer

    output = args.output.resolve()
    model_path = args.model.resolve()
    videos_dir = args.videos_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not videos_dir.is_dir():
        raise NotADirectoryError(f"video directory not found: {videos_dir}")
    videos = list(baseline_videos(args.baseline_transcripts, args.subset)[0])

    model_hash = sha256(model_path)
    print(f"loading model {model_path} (sha256 {model_hash})", flush=True)
    started = time.monotonic()
    recognizer = Recognizer.load(model_path)
    print(f"model loaded in {time.monotonic() - started:.1f} s", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        stream.flush()
        for position, video in enumerate(videos, 1):
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
            result = text if status == "ok" else detail
            print(f"{tag}: {status} in {time.monotonic() - started:.1f} s: {result}", flush=True)
    print(f"phase 5 done: {output} {dict(counts)}", flush=True)


def standardize_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    text = re.sub(r"\b(um|uh|hmm|mm|mhm|mmm)\b", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(first, second):
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
        if reader.fieldnames not in (TSV_COLUMNS, NO_HASH_TSV_COLUMNS):
            raise ValueError(f"{path}: not a phase 5 output")
        rows = {}
        for row in reader:
            if row["video"] in rows:
                raise ValueError(f"{path}: duplicate video {row['video']}")
            rows[row["video"]] = row
    if "model_sha256" in reader.fieldnames:
        models = {row["model_sha256"] for row in rows.values()}
        if len(models) > 1 or not all(models):
            raise ValueError(f"{path}: rows carry different or missing model hashes")
    return rows


def model_sha256(rows):
    """The model hash shared by the rows of read_purescribe, or None for a TSV without the column."""
    hashes = {row.get("model_sha256") for row in rows.values()}
    return hashes.pop() if len(hashes) == 1 else None


def distance_stats(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


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
        by_method[method] = {
            "n": len(group),
            "correct": correct,
            "accuracy": ratio(correct, len(group)),
            "distance": distance_stats([row[system]["distance"] for row in group]),
        }

    distance_by_ground_truth = {}
    for label in LABELS:
        distances = [row[system]["distance"] for row in rows if row["ground_truth"] == label]
        distance_by_ground_truth[label] = distance_stats(distances)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": ratio(tp + tn, len(rows)),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "distance_by_ground_truth": distance_by_ground_truth,
        "by_method": by_method,
    }


def avsr_export(args, videos):
    if args.avsr_manifest is None:
        raise ValueError("--avsr-transcripts needs --avsr-manifest")
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
    purescribe = read_purescribe(args.purescribe)
    unknown = sorted(purescribe.keys() - videos.keys())
    if unknown:
        raise ValueError(f"{args.purescribe}: videos without baseline transcripts: {unknown[:5]}")
    if args.avsr_transcripts is not None:
        manifest, avsr = avsr_export(args, videos)
    else:
        manifest, avsr = None, None

    rows = []
    excluded = Counter()
    digest = hashlib.sha256()
    excluded_avsr = Counter()
    excluded_avsr_videos = []
    for video, (whisper_txt, avsr_txt) in videos.items():
        status = purescribe[video]["status"] if video in purescribe else "not_transcribed"
        if status != "ok":
            excluded[status] += 1
            continue
        label, name = video.split("/", 1)
        if avsr is None:
            video_text = avsr_txt.read_text(encoding="utf-8")
        else:
            entry = avsr.get(video)
            if entry is None:
                raise ValueError(f"{args.avsr_manifest}: no entry for {video}")
            if entry["status"] != "ok":
                excluded_avsr[entry["status"]] += 1
                excluded_avsr_videos.append({"video": video, "status": entry["status"], "reason": entry.get("reason")})
                continue
            text_path = Path(args.avsr_transcripts) / "avsr" / args.subset / label / f"{name[:-4]}.txt"
            data = text_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["text_sha256"]:
                raise ValueError(f"{args.avsr_manifest}: text of {video} does not match its sha256")
            video_text = data.decode("utf-8")
        whisper_text = whisper_txt.read_text(encoding="utf-8")
        digest.update(f"{video}\0{whisper_text}\0{video_text}\0".encode("utf-8"))

        row = {
            "video": video,
            "ground_truth": label,
            "method": method_of(name, label),
            "video_transcription": video_text,
        }
        purescribe_text = standardize_text(purescribe[video]["purescribe_transcription"])
        for system, audio in (("whisper", whisper_text), ("purescribe", purescribe_text)):
            distance = normalized_levenshtein(audio, video_text)
            row[system] = {
                "audio_transcription": audio,
                "distance": distance,
                "prediction": "fake" if distance > args.threshold else "real",
            }
        rows.append(row)
    if not rows:
        raise ValueError("no video has both baseline transcripts and a PureScribe transcription")

    whisper = evaluate(rows, "whisper")
    pure = evaluate(rows, "purescribe")
    change = {}
    for key in ("accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"):
        if whisper[key] is None or pure[key] is None:
            change[key] = {"absolute": None, "relative": None}
        else:
            difference = pure[key] - whisper[key]
            change[key] = {"absolute": difference, "relative": ratio(difference, whisper[key])}

    result = {
        "phase": "6 compare",
        "inputs": {
            "baseline_transcripts": str(Path(args.baseline_transcripts).resolve()),
            "subset": args.subset,
            "baseline_texts_sha256": digest.hexdigest(),
            "purescribe_tsv": str(args.purescribe.resolve()),
            "purescribe_sha256": sha256(args.purescribe),
            "purescribe_model_sha256": model_sha256(purescribe),
        },
        "protocol": {
            "video_branch": "baseline lip-reading transcripts, unchanged",
            "audio_branch_baseline": "baseline Whisper transcripts, unchanged",
            "normalization": "baseline standardize_text: lower case, [..] and (..) removed, um/uh/hmm/mm/mhm/mmm "
                             "removed, ASCII punctuation removed, whitespace collapsed",
            "distance": "baseline normalized_levenshtein: Levenshtein / max length on characters, "
                        "1.0 if only one text is empty, rounded to 4 decimals",
            "threshold": args.threshold,
            "decision": "fake if distance > threshold",
            "positive_class": "fake",
        },
        "videos": {
            "baseline_videos": len(videos),
            "evaluated": len(rows),
            "excluded": dict(excluded),
            "baseline_incomplete": incomplete,
            "empty_video_transcriptions": sum(not row["video_transcription"].strip() for row in rows),
            "ground_truth": dict(Counter(row["ground_truth"] for row in rows)),
            "method": dict(Counter(row["method"] for row in rows)),
        },
        "whisper_baseline": whisper,
        "purescribe": pure,
        "purescribe_minus_whisper": change,
        "per_video": rows,
    }
    if avsr is not None:
        result["inputs"].update({
            "avsr_transcripts": str(Path(args.avsr_transcripts).resolve()),
            "avsr_manifest": str(Path(args.avsr_manifest).resolve()),
            "avsr_manifest_sha256": sha256(args.avsr_manifest),
            "avsr_checkpoint": manifest["parameters"]["checkpoint"],
            "avsr_preprocessing": manifest["parameters"]["preprocessing"],
        })
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
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lost in Translation with PureScribe as the audio branch.")
    phases = parser.add_subparsers(dest="phase", required=True)

    t = phases.add_parser("transcribe", help="phase 5: PureScribe transcription of the videos' audio")
    t.add_argument("--model", type=Path, required=True, help="PureScribe model (.psm) from phase 3")
    t.add_argument("--baseline-transcripts", type=Path, required=True,
                   help="baseline 'transcriptions' directory; only file names are read")
    t.add_argument("--subset", default="test")
    t.add_argument("--videos-dir", type=Path, required=True, help="directory with real/ and fake/ videos")
    t.add_argument("--output", type=Path, required=True, help="new TSV file (an existing one is refused)")
    t.set_defaults(func=transcribe)

    c = phases.add_parser("compare", help="phase 6: compare Whisper and PureScribe against lip reading")
    c.add_argument("--baseline-transcripts", type=Path, required=True)
    c.add_argument("--subset", default="test")
    c.add_argument("--purescribe", type=Path, required=True, help="TSV written by phase 5")
    c.add_argument("--threshold", type=threshold, default=0.5)
    c.add_argument("--output", type=Path, required=True, help="new JSON file")
    c.add_argument("--avsr-transcripts", type=Path,
                   help="output root of a new video-only Auto-AVSR export, instead of the baseline lip-reading texts")
    c.add_argument("--avsr-manifest", type=Path, help="manifest.json of that export (required with --avsr-transcripts)")
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
