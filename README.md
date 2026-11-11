# PureScribe

PureScribe trascrive file audio usando un modello addestrato su parole note. Per addestrare un modello si realizza una cartella per ogni parola con all'interno i relativi file WAV. Il programma salva il modello in un file `.psm`.

Il riconoscitore usa MFCC per descrivere l'audio e modelli HMM-GMM per scegliere le parole, dopodiché usa PureScribe per trascrivere l'audio di video e confrontare il risultato con trascrizioni già disponibili.

## Requisiti e installazione

Per eseguire tutta la pipeline su Windows servono Python 3.10–3.13 e FFmpeg.
scarica [Python per Windows](https://www.python.org/downloads/windows/) e FFmpeg dalla [pagina ufficiale di download](https://ffmpeg.org/download.html) oppure tramite winget se disponibile.

Dopo l'installazione, chiudi e riapri cmd. Controlla che Python sia tra le versioni 3.10 e 3.13 e che FFmpeg sia disponibile:

```cmd
python --version
ffmpeg -version
```

Poi, dalla cartella principale del progetto, installa i pacchetti Python:

```cmd
python -m pip install -e ".[pipeline]"
```

`pyproject.toml` contiene le dipendenze che ti servono: installa PureScribe, NumPy, SciPy, WhisperX 3.8.6 e i pacchetti python richiesti.


## Procedimenti
### 2. Clip Common Voice

WhisperX trova i tempi delle parole nell'audio. Il testo ufficiale Common Voice fornisce le etichette dei clip. Il risultato è una cartella con sottocartelle, una per parola, e file WAV.

```cmd
python commonvoice_purescribe.py extract --data-root corpus --audio-dir <corpus\validated> --transcription-file <corpus\validated.tsv> --output .\validated_new
```

Scegli una nuova cartella di output.

### 3. Addestrare PureScribe

```cmd
python -m purescribe.cli train .\validated_new -o .\model_new.psm
```

La cartella di ingresso deve contenere una cartella per parola con i relativi WAV.

### 5. Trascrivere i video

La cartella dei video deve avere le sottocartelle `real` e `fake`. La trascrizione usa l'audio dei video e salva i risultati in un TSV.

```cmd
python lost_in_translation.py transcribe --model .\model_new.psm --baseline-transcripts <\percorso\transcriptions> --subset test --videos-dir <\percorso\videos> --output .\run_new.tsv
```

### 6. Confrontare i risultati

Il confronto usa testi già pronti: la trascrizione audio di Whisper e quella del labiale. Confronta anche il testo di PureScribe con quello del labiale e salva i risultati in un JSON. 

```cmd
python lost_in_translation.py compare --baseline-transcripts <\percorso\transcriptions> --subset test --purescribe .\run_new.tsv --threshold 0.5 --output .\metrics_new.json
```

La soglia predefinita è `0.5`. Puoi indicare un'altra soglia con `--threshold`.

## Comandi PureScribe

```cmd
python -m purescribe.cli --help
python -m purescribe.cli train .\cartella_wav -o .\model_new.psm
python -m purescribe.cli transcribe .\audio.wav -m .\model_new.psm
python -m purescribe.cli classify .\audio.wav -m .\model_new.psm
python -m purescribe.cli info .\model_new.psm
python -m purescribe.cli features .\audio.wav
```

Sostituisci `cartella_wav` e `audio.wav` con i tuoi percorsi.

