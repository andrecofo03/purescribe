from __future__ import annotations

import math
import re
import wave
from collections import defaultdict
from difflib import SequenceMatcher
from numbers import Real
from pathlib import Path

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
DATA_DIR = Path(r"D:/CommonVoice")
OUTPUT_DIR = Path(r"C:/Users/cofon/Desktop/words_cv")
TARGET_SR = 16000  # Campionamento atteso da PureScribe

# Numero massimo di registrazioni da salvare per ciascuna parola
CLIPS_PER_WORD = 25

PADDING_START_SEC = 0.050
PADDING_END_SEC = 0.120
MIN_CLIP_DURATION_SEC = 0.100

# VOCABOLARIO TARGET:
# Se impostato su None, estrarrà le prime parole valide che incontra fino a capienza.
# Altrimenti definisci un set, es: {"about", "above", "call", "letter"}
TARGET_VOCABULARY = None
# ==============================================================================


def clean_word(word: str) -> str:
    """Rimuove punteggiatura e converte in minuscolo."""
    return re.sub(r"[^\w]", "", word.lower().strip())


def _comparison_word(word: str) -> str:
    word = word.casefold().replace("’", "'").replace("‘", "'")
    return re.sub(r"^[\W_]+|[\W_]+$", "", word)


def match_word_indices(
    reference_words: list[str], whisper_words: list[str]
) -> list[tuple[int, int]]:
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


def alignment_statistics(
    reference_words: list[str],
    whisper_words: list[str],
    matches: list[tuple[int, int]],
    target_vocabulary: set[str] | None,
) -> dict[str, int]:
    target_indices = {
        index
        for index, word in enumerate(reference_words)
        if target_vocabulary is None or word in target_vocabulary
    }
    target_matches = [
        (reference_index, whisper_index)
        for reference_index, whisper_index in matches
        if reference_index in target_indices
    ]
    matched_reference_indices = {
        reference_index for reference_index, _ in target_matches
    }
    matched_whisper_indices = {whisper_index for _, whisper_index in target_matches}
    return {
        "commonvoice_words_considered": len(target_indices),
        "exact_matches": len(matched_reference_indices),
        "reference_words_discarded": len(target_indices)
        - len(matched_reference_indices),
        "whisper_words_ignored": len(whisper_words) - len(matched_whisper_indices),
    }


def valid_word_bounds(start, end, audio_duration: float) -> tuple[float, float] | None:
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


def ensure_word_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, NotADirectoryError) as exc:
        if path.is_dir():
            raise
        raise NotADirectoryError(
            f"Il percorso di output non è una cartella: {path}"
        ) from exc
    if not path.is_dir():
        raise NotADirectoryError(f"Il percorso di output non è una cartella: {path}")


def save_wav(path: Path, audio: np.ndarray, sr: int):
    """Salva l'array come WAV mono 16-bit PCM."""
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


def main():
    import librosa
    import pandas as pd
    import stable_whisper
    import torch

    tsv_path = DATA_DIR / "validated.tsv"
    clips_dir = DATA_DIR / "validated"

    if not tsv_path.exists():
        raise FileNotFoundError(f"File non trovato: {tsv_path}")

    print(f"Lettura manifest: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inizializzazione Stable Whisper (modello 'base' su {device.upper()})...")
    whisper_model = stable_whisper.load_model("base", device=device)

    stats = {
        "manifest_rows_examined": 0,
        "audio_missing": 0,
        "load_errors": 0,
        "whisper_errors": 0,
        "invalid_rows": 0,
        "commonvoice_words_considered": 0,
        "whisper_words_obtained": 0,
        "exact_matches": 0,
        "reference_words_discarded": 0,
        "whisper_words_ignored": 0,
        "clips_saved": 0,
        "clips_skipped_limit": 0,
        "invalid_timestamps": 0,
        "clip_save_errors": 0,
    }
    word_counts = defaultdict(int)
    next_clip_indices = defaultdict(int)

    if OUTPUT_DIR.is_dir():
        for word_dir in OUTPUT_DIR.iterdir():
            if not word_dir.is_dir():
                continue
            existing_clips = [
                path
                for path in word_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".wav"
            ]
            word_counts[word_dir.name] = len(existing_clips)
            index = 0
            while (word_dir / f"{word_dir.name}_{index:04d}.wav").exists():
                index += 1
            next_clip_indices[word_dir.name] = index

    for idx, row in df.iterrows():
        if TARGET_VOCABULARY is not None and all(
            word_counts[w] >= CLIPS_PER_WORD for w in TARGET_VOCABULARY
        ):
            print("\nRaggiunto il numero di campioni per tutte le parole del vocabolario!")
            break
        stats["manifest_rows_examined"] += 1

        path_value = row.get("path")
        raw_sentence = row.get("sentence")
        if (
            not isinstance(path_value, str)
            or not path_value.strip()
            or not isinstance(raw_sentence, str)
        ):
            stats["invalid_rows"] += 1
            print(f"[!] Riga {idx + 1}: campione non valido nel manifest")
            continue

        reference_entries = [
            (source, normalized)
            for source in raw_sentence.split()
            if (normalized := clean_word(source))
        ]
        reference_source_words = [source for source, _ in reference_entries]
        ref_words = [normalized for _, normalized in reference_entries]
        if not ref_words:
            stats["invalid_rows"] += 1
            print(f"[!] Riga {idx + 1}: nessuna parola valida nella trascrizione")
            continue

        needed = any(
            (TARGET_VOCABULARY is None or word in TARGET_VOCABULARY)
            and word_counts[word] < CLIPS_PER_WORD
            for word in ref_words
        )
        if not needed:
            continue

        rejected_sample_stats = alignment_statistics(
            ref_words, [], [], TARGET_VOCABULARY
        )
        stats["commonvoice_words_considered"] += rejected_sample_stats[
            "commonvoice_words_considered"
        ]
        audio_path = clips_dir / path_value
        if not audio_path.is_file():
            stats["audio_missing"] += 1
            stats["reference_words_discarded"] += rejected_sample_stats[
                "reference_words_discarded"
            ]
            print(f"[!] Riga {idx + 1}: audio mancante ({path_value})")
            continue

        try:
            audio_array, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        except Exception as exc:
            stats["load_errors"] += 1
            stats["reference_words_discarded"] += rejected_sample_stats[
                "reference_words_discarded"
            ]
            print(f"[!] Riga {idx + 1}: errore caricamento audio ({type(exc).__name__})")
            continue

        try:
            result = whisper_model.transcribe(
                audio_array,
                language="en",
                word_timestamps=True,
                # Stable TS 2.19.1 può comprimere la prima parola nella pausa iniziale.
                suppress_silence=False,
            )
            words_list = result.all_words()
            whisper_entries = [
                (word.word, normalized, word)
                for word in words_list
                if (normalized := clean_word(word.word))
            ]
        except Exception as exc:
            stats["whisper_errors"] += 1
            stats["reference_words_discarded"] += rejected_sample_stats[
                "reference_words_discarded"
            ]
            print(f"[!] Riga {idx + 1}: errore Whisper ({type(exc).__name__})")
            continue

        whisper_source_words = [source for source, _, _ in whisper_entries]
        whisper_words = [normalized for _, normalized, _ in whisper_entries]
        stats["whisper_words_obtained"] += len(whisper_words)
        matches = match_word_indices(reference_source_words, whisper_source_words)
        sample_stats = alignment_statistics(
            ref_words, whisper_words, matches, TARGET_VOCABULARY
        )
        for key in (
            "exact_matches",
            "reference_words_discarded",
            "whisper_words_ignored",
        ):
            stats[key] += sample_stats[key]
        audio_duration = len(audio_array) / sr

        for reference_index, whisper_index in matches:
            target = ref_words[reference_index]
            if TARGET_VOCABULARY is not None and target not in TARGET_VOCABULARY:
                continue

            if word_counts[target] >= CLIPS_PER_WORD:
                stats["clips_skipped_limit"] += 1
                continue

            whisper_word = whisper_entries[whisper_index][2]
            bounds = valid_word_bounds(
                getattr(whisper_word, "start", None),
                getattr(whisper_word, "end", None),
                audio_duration,
            )
            if bounds is None:
                stats["invalid_timestamps"] += 1
                print(f"[!] Riga {idx + 1}: timestamp Whisper non validi per '{target}'")
                continue
            start, end = bounds

            start_sec = max(0.0, start - PADDING_START_SEC)
            end_sec = min(audio_duration, end + PADDING_END_SEC)
            clip_audio = audio_array[int(start_sec * sr) : int(end_sec * sr)]
            if len(clip_audio) < int(MIN_CLIP_DURATION_SEC * sr):
                continue

            word_dir = OUTPUT_DIR / target
            try:
                ensure_word_directory(word_dir)
            except Exception as exc:
                stats["clip_save_errors"] += 1
                print(
                    f"[!] Riga {idx + 1}: cartella di output non valida "
                    f"'{word_dir}' ({type(exc).__name__})"
                )
                continue

            clip_index = next_clip_indices[target]
            saved = False
            while not saved:
                out_path = word_dir / f"{target}_{clip_index:04d}.wav"
                try:
                    save_wav(out_path, clip_audio, sr)
                except FileExistsError:
                    if out_path.is_file():
                        clip_index += 1
                        next_clip_indices[target] = clip_index
                    else:
                        stats["clip_save_errors"] += 1
                        print(
                            f"[!] Riga {idx + 1}: percorso WAV non valido "
                            f"'{out_path}' (FileExistsError)"
                        )
                        break
                except Exception as exc:
                    stats["clip_save_errors"] += 1
                    print(
                        f"[!] Riga {idx + 1}: errore salvataggio '{target}' "
                        f"({type(exc).__name__})"
                    )
                    break
                else:
                    saved = True
                    word_counts[target] += 1
                    next_clip_indices[target] = clip_index + 1
                    stats["clips_saved"] += 1
                    dur = end_sec - start_sec
                    print(
                        f"  [+] '{target}' estratta ({start_sec:.2f}s - "
                        f"{end_sec:.2f}s, dur: {dur:.2f}s)"
                    )

    print("\nEstrazione completata!")
    print(f"Righe del manifest esaminate: {stats['manifest_rows_examined']}")
    print(f"Audio mancanti: {stats['audio_missing']}")
    print(f"Errori di caricamento: {stats['load_errors']}")
    print(f"Errori Whisper: {stats['whisper_errors']}")
    print(f"Righe non valide: {stats['invalid_rows']}")
    print(f"Parole Common Voice considerate: {stats['commonvoice_words_considered']}")
    print(f"Parole Whisper ottenute: {stats['whisper_words_obtained']}")
    print(f"Corrispondenze esatte: {stats['exact_matches']}")
    print(f"Parole di riferimento scartate: {stats['reference_words_discarded']}")
    print(f"Parole Whisper ignorate: {stats['whisper_words_ignored']}")
    print(f"Clip salvati: {stats['clips_saved']}")
    print(f"Clip non salvati per limite raggiunto: {stats['clips_skipped_limit']}")
    print(f"Timestamp Whisper non validi: {stats['invalid_timestamps']}")
    print(f"Errori di salvataggio clip: {stats['clip_save_errors']}")
    coverage = (
        stats["exact_matches"] / stats["commonvoice_words_considered"] * 100
        if stats["commonvoice_words_considered"]
        else 0.0
    )
    print(f"Copertura: {coverage:.1f}%")
    for word, count in sorted(word_counts.items(), key=lambda x: str(x[0])):
        print(f"  {str(word):<15}: {count} file .wav")


if __name__ == "__main__":
    main()
