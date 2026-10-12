from pathlib import Path
import argparse
import random
import wave

import numpy as np


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        n = wf.getnframes()
        data = wf.readframes(n)

    if width == 2:
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float64) / 32768.0
    else:
        raise ValueError(f"Only 16-bit WAV supported: {path}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return rate, audio


def write_wav(path, audio, rate):
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


def make_sequences(words_dir, output_dir, n_sequences=40,
                   min_words=3, max_words=5, seed=None):

    rng = random.Random(seed)

    words_dir = Path(words_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocabulary = sorted(
        p.name for p in words_dir.iterdir()
        if p.is_dir()
    )

    if not vocabulary:
        raise RuntimeError(f"No word directories found in {words_dir}")

    print("Vocabulary:", vocabulary)

    word_files = {}

    for word in vocabulary:
        files = sorted((words_dir / word).glob("*.wav"))

        if not files:
            raise RuntimeError(f"No WAV files found for word: {word}")

        word_files[word] = files

    references = []

    for i in range(n_sequences):
        n_words = rng.randint(min_words, max_words)
        words = [rng.choice(vocabulary) for _ in range(n_words)]

        pieces = []

        rate = None
        pieces.append(None)

        for word in words:
            wav = rng.choice(word_files[word])

            wav_rate, audio = read_wav(wav)

            if rate is None:
                rate = wav_rate
            elif wav_rate != rate:
                raise ValueError(
                    f"Sample-rate mismatch: {wav} has {wav_rate}, "
                    f"expected {rate}"
                )

            pieces.append(audio)

            pause_duration = rng.uniform(0.12, 0.30)
            pause = np.zeros(int(pause_duration * rate))
            pieces.append(pause)
        initial = np.zeros(int(0.25 * rate))
        pieces[0] = initial

        audio = np.concatenate(pieces)

        filename = f"seq_{i:03d}.wav"
        write_wav(output_dir / filename, audio, rate)

        references.append(
            f"{filename}\t{' '.join(words)}"
        )

    (output_dir / "reference.txt").write_text(
        "\n".join(references) + "\n"
    )

    print(f"Created {n_sequences} sequences")
    print(f"Output: {output_dir}")
    print(f"Reference: {output_dir / 'reference.txt'}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "words",
        help="Directory containing one subdirectory per word"
    )

    parser.add_argument(
        "output",
        help="Directory where sequences will be written"
    )

    parser.add_argument(
        "--sequences",
        type=int,
        default=40
    )

    parser.add_argument(
        "--min-words",
        type=int,
        default=3
    )

    parser.add_argument(
        "--max-words",
        type=int,
        default=5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    args = parser.parse_args()

    make_sequences(
        args.words,
        args.output,
        args.sequences,
        args.min_words,
        args.max_words,
        args.seed,
    )


if __name__ == "__main__":
    main()
