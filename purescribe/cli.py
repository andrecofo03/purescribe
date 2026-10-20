"""Command line interface: purescribe train | transcribe | classify | info."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def _cmd_train(args) -> int:
    if Path(args.output).exists():
        raise FileExistsError(f"{args.output}; choose a new model filename")
    from .features import FeatureConfig
    from .train import TrainConfig, train_from_directory

    fcfg = FeatureConfig(rate=args.rate, n_cepstra=args.cepstra,
                         fmax=args.fmax, cmvn_mode=args.cmvn)
    tcfg = TrainConfig(
        n_mixtures=args.mixtures,
        iterations=args.iterations,
        method="baum_welch" if args.baum_welch else "viterbi",
    )
    rec = train_from_directory(args.corpus, fcfg, tcfg, verbose=not args.quiet)

    if args.grammar:
        sentences = [line.split() for line in Path(args.grammar).read_text().splitlines() if line.strip()]
        rec.lexicon.fit(sentences)
        rec._decoder = None
        if not args.quiet:
            print(f"bigram grammar fitted on {len(sentences)} sentences")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete model exclusively, including if another run finished meanwhile.
    with tempfile.TemporaryDirectory(prefix=".purescribe-train-", dir=out.parent) as staging:
        staged = rec.save(Path(staging) / "model.psm")
        if sys.platform == "win32":
            staged.rename(out)  # Windows rename refuses an existing destination.
        else:
            os.link(staged, out)
    print(f"saved {out}  ({out.stat().st_size / 1024:.1f} kB, "
          f"vocabulary {len(rec.lexicon)})")
    return 0


def _cmd_transcribe(args) -> int:
    from .recognizer import Recognizer

    rec = Recognizer.load(args.model)
    use_vad = None
    if args.no_vad:
        use_vad = False
    elif args.vad:
        use_vad = True
    if args.lm_scale is not None:
        rec.decode_config.lm_scale = args.lm_scale
    if args.word_penalty is not None:
        rec.decode_config.word_penalty = args.word_penalty
    rec._decoder = None

    for path in args.audio:
        transcript = rec.transcribe(path, use_vad=use_vad)
        if args.format == "json":
            print(json.dumps({"audio": str(path), **transcript.to_dict()}, indent=2))
        elif args.format == "srt":
            print(transcript.to_srt())
        elif args.format == "timed":
            for w in transcript.words:
                print(f"{w.start:8.2f} {w.end:8.2f}  {w.text}")
        else:
            if len(args.audio) > 1:
                print(f"{path}: {transcript.text}")
            else:
                print(transcript.text)
    return 0


def _cmd_classify(args) -> int:
    from .recognizer import Recognizer

    rec = Recognizer.load(args.model)
    for path in args.audio:
        word, score = rec.classify(path)
        print(f"{path}: {word}  (log-likelihood {score:.1f})")
    return 0


def _cmd_info(args) -> int:
    from .recognizer import Recognizer

    rec = Recognizer.load(args.model)
    fc = rec.feature_config
    print(f"model            {args.model}")
    print(f"vocabulary       {len(rec.lexicon)} words: {', '.join(rec.vocabulary)}")
    print(f"features         {fc.n_features}-dim "
          f"({fc.n_cepstra} MFCC{' + delta' if fc.deltas else ''}"
          f"{' + delta-delta' if fc.delta_deltas else ''})")
    print(f"front end        {fc.rate} Hz, {fc.frame_ms:g} ms frame / "
          f"{fc.hop_ms:g} ms hop, {fc.n_filters} mel filters")
    print(f"grammar          {'bigram' if rec.lexicon.bigram else 'uniform unigram'}")
    print(f"decoding         lm_scale {rec.decode_config.lm_scale:g}, "
          f"word_penalty {rec.decode_config.word_penalty:g}")
    print("acoustic models:")
    for name in sorted(rec.models):
        m = rec.models[name]
        params = sum(g.means.size + g.variances.size + g.weights.size for g in m.states)
        print(f"  {name:<20s} {m.n_states:2d} states x {m.n_mixtures} mix "
              f"({params:,} parameters)")
    return 0


def _cmd_features(args) -> int:
    from .audio import load_audio
    from .features import FeatureConfig, mfcc
    from .vad import detect_speech

    wave = load_audio(args.audio)
    feats = mfcc(wave.samples, FeatureConfig(rate=wave.rate))
    segs = detect_speech(wave.samples, wave.rate)
    print(f"{args.audio}")
    print(f"  duration     {wave.duration:.2f} s @ {wave.rate} Hz")
    print(f"  features     {feats.shape[0]} frames x {feats.shape[1]} dims")
    print(f"  speech       {len(segs)} segment(s), "
          f"{sum(s.duration for s in segs):.2f} s active")
    for s in segs:
        print(f"    {s.start:7.2f} - {s.end:7.2f} s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="purescribe",
        description="Classical DSP speech recognition. No neural networks, no LLMs.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train acoustic models from a labelled corpus")
    t.add_argument("corpus", help="directory laid out as <corpus>/<word>/*.wav")
    t.add_argument("-o", "--output", default="model.psm")
    t.add_argument("-m", "--mixtures", type=int, default=2,
                   help="Gaussians per state (default 2)")
    t.add_argument("-n", "--iterations", type=int, default=10)
    t.add_argument("--rate", type=int, default=16000)
    t.add_argument("--cepstra", type=int, default=13)
    t.add_argument("--fmax", type=float, default=None,
                   help="upper filterbank edge in Hz (default Nyquist). Lower it "
                        "to about 4000 when decoding lossy or telephone audio: "
                        "low-bitrate codecs discard everything above ~6 kHz, and "
                        "models that rely on that band fail outright on it.")
    t.add_argument("--cmvn", choices=["global", "utterance", "none"], default="global",
                   help="cepstral normalisation scope (default global)")
    t.add_argument("--baum-welch", action="store_true",
                   help="soft EM instead of segmental k-means")
    t.add_argument("--grammar", help="text file of example sentences for the bigram")
    t.add_argument("-q", "--quiet", action="store_true")
    t.set_defaults(func=_cmd_train)

    r = sub.add_parser("transcribe", help="transcribe audio with a trained model")
    r.add_argument("audio", nargs="+")
    r.add_argument("-m", "--model", default="model.psm")
    r.add_argument("-f", "--format", choices=["text", "timed", "json", "srt"], default="text")
    r.add_argument("--no-vad", action="store_true",
                   help="always decode the whole file in one pass")
    r.add_argument("--vad", action="store_true",
                   help="always pre-segment on silence (default: only if long)")
    r.add_argument("--lm-scale", type=float)
    r.add_argument("--word-penalty", type=float)
    r.set_defaults(func=_cmd_transcribe)

    c = sub.add_parser("classify", help="single-word classification of a whole clip")
    c.add_argument("audio", nargs="+")
    c.add_argument("-m", "--model", default="model.psm")
    c.set_defaults(func=_cmd_classify)

    i = sub.add_parser("info", help="describe a trained model")
    i.add_argument("model")
    i.set_defaults(func=_cmd_info)

    f = sub.add_parser("features", help="inspect the front end on one file")
    f.add_argument("audio")
    f.set_defaults(func=_cmd_features)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileExistsError as e:
        print(f"purescribe: output already exists: {e}", file=sys.stderr)
        return 2
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"purescribe: not found: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"purescribe: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
