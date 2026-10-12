import re
import wave
from collections import defaultdict
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import torch
import stable_whisper

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
DATA_DIR = Path(r"D:/CommonVoice")
OUTPUT_DIR = Path(r"C:/Users/cofon/Desktop/words_cv")
TARGET_SR = 16000  # Campionamento atteso da PureScribe

# Numero massimo di registrazioni da salvare per ciascuna parola
CLIPS_PER_WORD = 25

# Margine di sicurezza sui bordi del taglio (25 ms) per non troncare le consonanti
PADDING_SEC = 0.025

# VOCABOLARIO TARGET:
# Se impostato su None, estrarrà le prime parole valide che incontra fino a capienza.
# Altrimenti definisci un set, es: {"about", "above", "call", "letter"}
TARGET_VOCABULARY = None
# ==============================================================================


def clean_word(word: str) -> str:
    """Rimuove punteggiatura e converte in minuscolo."""
    return re.sub(r"[^\w]", "", word.lower().strip())


def save_wav(path: Path, audio: np.ndarray, sr: int):
    """Salva l'array come WAV mono 16-bit PCM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inizializzazione Stable Whisper (modello 'base' su {device.upper()})...")
    whisper_model = stable_whisper.load_model("base", device=device)

    tsv_path = DATA_DIR / "validated.tsv"  # oppure validated.tsv / train.tsv se preferisci
    clips_dir = DATA_DIR / "validated"

    if not tsv_path.exists():
        raise FileNotFoundError(f"File non trovato: {tsv_path}")

    print(f"Lettura manifest: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")

    word_counts = defaultdict(int)

    for idx, row in df.iterrows():
        # Controllo se abbiamo già completato tutte le parole target
        if TARGET_VOCABULARY and all(
            word_counts[w] >= CLIPS_PER_WORD for w in TARGET_VOCABULARY
        ):
            print("\nRaggiunto il numero di campioni per tutte le parole del vocabolario!")
            break

        audio_path = clips_dir / row["path"]
        if not audio_path.exists():
            continue

        raw_sentence = str(row["sentence"])
        ref_words = [clean_word(w) for w in raw_sentence.split() if clean_word(w)]
        if not ref_words:
            continue

        # Verifica preliminare: la frase contiene almeno una parola che ci serve?
        if TARGET_VOCABULARY:
            needed = any(
                w in TARGET_VOCABULARY and word_counts[w] < CLIPS_PER_WORD
                for w in ref_words
            )
            if not needed:
                continue

        # 1. Caricamento audio con librosa a 16 kHz mono
        try:
            audio_array, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        except Exception:
            continue

        # 2. Allineamento forzato direttamente sul testo ufficiale di Common Voice
        try:
            result = whisper_model.align(audio_array, raw_sentence, language="en")
            words_list = result.all_words()
        except Exception:
            continue

        # 3. Estrazione delle parole allineate
        for w in words_list:
            target = clean_word(w.word)
            if not target:
                continue

            is_target = TARGET_VOCABULARY is None or target in TARGET_VOCABULARY
            if is_target and word_counts[target] < CLIPS_PER_WORD:
                # Margine iniziale per catturare l'attacco della consonante (50 ms)
                start_sec = max(0.0, w.start - 0.050)
                # Margine finale generoso (+120 ms) per catturare l'intera sillaba di chiusura e rilascio
                end_sec = min(len(audio_array) / sr, w.end + 0.120)

                clip_audio = audio_array[int(start_sec * sr) : int(end_sec * sr)]

                # Scarta ritagli anomali inferiori a 100 ms
                if len(clip_audio) >= int(0.10 * sr):
                    c_idx = word_counts[target]
                    out_path = OUTPUT_DIR / target / f"{target}_{c_idx:04d}.wav"
                    save_wav(out_path, clip_audio, sr)
                    word_counts[target] += 1
                    dur = end_sec - start_sec
                    print(f"  [+] '{target}' estratta ({start_sec:.2f}s - {end_sec:.2f}s, dur: {dur:.2f}s)")

        if (idx + 1) % 25 == 0:
            total_extracted = sum(word_counts.values())
            print(
                f"Clip esaminate: {idx + 1} | Parole estratte totali: {total_extracted}"
            )

    print("\nEstrazione completata!")
    for word, count in sorted(word_counts.items(), key=lambda x: str(x[0])):
        if word is not None:
            print(f"  {str(word):<15}: {count} file .wav")


if __name__ == "__main__":
    main()