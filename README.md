# PureScribe

Riconoscimento vocale classico: MFCC, modelli acustici HMM-GMM e decodifica Viterbi.
Il vocabolario è finito e coincide con le cartelle del corpus di training. Nessuna rete neurale.

In questa tesi PureScribe sostituisce Whisper come trascrittore audio nella baseline
"Lost in Translation" (Bohacek e Farid, CVPRW 2024). Il metodo confronta la trascrizione
dell'audio con quella del labiale (lip reading) usando la distanza di Levenshtein normalizzata
per riconoscere i lip-sync deepfake.

## Pipeline

| Fase | Comando | Dove |
|---|---|---|
| 2 estrazione delle parole con Whisper | `commonvoice_purescribe.py extract` | offline, Common Voice `dev` |
| 3 addestramento PureScribe | `python -m purescribe.cli train` | offline |
| 5 trascrizione audio dei video con PureScribe | `lost_in_translation.py transcribe` | pipeline finale |
| 6 confronto con la trascrizione del labiale | `lost_in_translation.py compare` | pipeline finale |

Le fasi 1 (split train/validation di Common Voice) e 4 (validation su Common Voice) sono state
rimosse: tutti i clip Common Voice servono all'addestramento e il sistema viene valutato a valle
su DeepSpeak (`D:\DeepSpeak\test`). La numerazione delle altre fasi è invariata.

Whisper serve solo nella fase 2, per ricavare timestamp candidati sulle parole di training.
Le etichette dei clip sono sempre le parole ufficiali Common Voice. Le fasi 5–6 non eseguono
Whisper: PureScribe trascrive l'audio dei video e le metriche della baseline Whisper vengono
ricalcolate, con il protocollo della baseline, dalle trascrizioni che la baseline aveva già
prodotto (`C:\Users\cofon\DeepSpeak\transcriptions`, esecuzione del 20/09/2026 su `D:\DeepSpeak\test`).

Comandi da PowerShell, nella root del repository; tutti i file prodotti restano nella root:

```powershell
# fase 2: la prima volta i percorsi si passano esplicitamente, poi vengono ricordati;
# rilanciando, l'estrazione precedente in corpus_cv viene sovrascritta
.\.venv\Scripts\python.exe commonvoice_purescribe.py extract `
    --data-root corpus `
    --audio-dir corpus\dev `
    --transcription-file corpus\dev.tsv `
    --output corpus_cv `
    --clips-per-word 23 `
    --min-clips-per-word 12 `
    --whisper-model base

# fase 3
.\.venv\Scripts\python.exe -m purescribe.cli train corpus_cv -o purescribe_cv.psm

# fase 5
.\.venv\Scripts\python.exe lost_in_translation.py transcribe `
    --model purescribe_cv.psm `
    --baseline-transcripts C:\Users\cofon\DeepSpeak\transcriptions `
    --subset test `
    --videos-dir D:\DeepSpeak\test `
    --output deepfake_purescribe.tsv
# (dopo un'interruzione: stesso comando con --resume; salta i video già completati)

# fase 6
.\.venv\Scripts\python.exe lost_in_translation.py compare `
    --baseline-transcripts C:\Users\cofon\DeepSpeak\transcriptions `
    --subset test `
    --purescribe deepfake_purescribe.tsv `
    --threshold 0.5 `
    --output deepfake_comparison.json
```

`corpus\` del repository contiene i dati Common Voice di ingresso: `dev\*.mp3`, `dev.tsv`
(frasi ufficiali e metadati di ogni audio) e `words_cv\` (parole singole già estratte dagli audio
di `dev`). È protetta: non usarla come `--output` (l'estrazione la rifiuta, perché contiene gli
input e non un'estrazione precedente).

Le fasi 3, 5 e 6 rifiutano output esistenti (la fase 5 li continua solo con `--resume`, dopo
aver verificato intestazione, ordine dei video e hash del modello; mostra indice, durata, esito e
tempo di ogni video; circa 2,6 s di calcolo per secondo di audio); la fase 2 sovrascrive solo una propria estrazione
precedente nella stessa cartella (altre cartelle vengono rifiutate). La fase 2 elabora tutti gli audio di `dev`
(nessun limite): circa 0,8 s per audio su CPU nel pilota precedente, quindi ~3,6 ore per i 16.402
audio utilizzabili; Ctrl+C la ferma in anticipo e il corpus raccolto viene comunque potato e
riassunto. Predefiniti: `--clips-per-word 23`, `--min-clips-per-word 12`, `--whisper-model base`,
`--output corpus_cv`. I percorsi usati sono salvati in `commonvoice_purescribe.settings.json`. Requisiti: l'ambiente `.venv` già
predisposto (numpy, scipy, openai-whisper) e ffmpeg nel PATH.

## Documentazione

- [Pipeline: obiettivo, ruoli, fasi, metriche, limiti](docs/pipeline.md)
- [Common Voice: sorgenti, vocabolario dinamico, matching, provenienza](docs/commonvoice.md)

## Contenuto del repository

| Percorso | Ruolo |
|---|---|
| `commonvoice_purescribe.py` | fase 2: estrazione con Whisper del corpus di training; funzioni di scoring WER |
| `purescribe/` | riconoscitore (MFCC, HMM-GMM, Viterbi) e CLI `train`/`transcribe`/`info` (fase 3) |
| `lost_in_translation.py` | fasi 5, 6: trascrizione dei video con PureScribe e confronto con il labiale |
| `tests/` | test del software con audio sintetico |
| `docs/`, `thesis/`, `verify/` | documentazione, materiali della tesi, registri delle verifiche |

`words.psm`, `corpus/` e `temp_iso.wav` sono dati preesistenti protetti: la pipeline non li usa.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp $env:TEMP\purescribe_pytest
```

I test verificano matching delle parole, validazione dei timestamp, scrittura dei WAV, lettura
delle trascrizioni ufficiali, estrazione con quote e potatura (Whisper simulato), calcolo della WER, esecuzione delle CLI `train` e `transcribe`
e protezione del modello in `train`, l'equivalenza bit per bit del decoder con la versione
precedente (`tests/reference_decoder.py`) e la ripresa `--resume` della fase 5.
Non misurano l'accuratezza su voce reale.
