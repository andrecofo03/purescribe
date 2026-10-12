from pathlib import Path
import argparse
from purescribe import Recognizer


def transcribe_sequences(model_path, sequences_dir, output_file):
    model_path = Path(model_path)
    sequences_dir = Path(sequences_dir)
    output_file = Path(output_file)

    if not model_path.exists():
        raise FileNotFoundError(f"File del modello non trovato: {model_path.resolve()}")
    if not sequences_dir.exists():
        raise FileNotFoundError(f"Cartella sequenze non trovata: {sequences_dir.resolve()}")

    print(f"Caricamento modello da: {model_path}")
    rec = Recognizer.load(str(model_path))

    wav_files = sorted(sequences_dir.glob("seq_*.wav"))
    if not wav_files:
        raise RuntimeError(f"Nessun file seq_*.wav trovato in: {sequences_dir}")

    lines = []
    print(f"Trascrizione in corso ({len(wav_files)} sequenze)...")

    for wav_path in wav_files:
        transcription = rec.transcribe(wav_path).text.strip()
        lines.append(f"{wav_path.name}\t{transcription}")
        print(f"  {wav_path.name} -> {transcription}")

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSalvate {len(lines)} trascrizioni in: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Trascrive le sequenze audio tramite Recognizer.")
    parser.add_argument("model", help="Percorso del modello addestrato (.psm)")
    parser.add_argument("sequences_dir", help="Cartella contenente i file seq_*.wav")
    parser.add_argument("-o", "--output", default="predictions.txt", help="File di output")

    args = parser.parse_args()
    transcribe_sequences(args.model, args.sequences_dir, args.output)


if __name__ == "__main__":
    main()