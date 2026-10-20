# Pipeline della tesi: PureScribe al posto di Whisper in "Lost in Translation"

## Obiettivo

"Lost in Translation" (Bohacek e Farid, CVPRW 2024) rileva i lip-sync deepfake confrontando
due trascrizioni indipendenti dello stesso video: quella dell'audio (Whisper) e quella del
labiale (lip reading, Auto-AVSR). Le trascrizioni sono confrontate con una distanza di
Levenshtein normalizzata; oltre la soglia 0,5 il video è classificato come deepfake.

La tesi mantiene il metodo e sostituisce il trascrittore audio con **PureScribe**, un
riconoscitore classico (MFCC, HMM-GMM, Viterbi) addestrato su parole di Common Voice.
Domanda sperimentale: a parità di video, trascrizione del labiale, normalizzazione, metrica,
soglia e protocollo, come cambiano le prestazioni passando da Whisper a PureScribe?

## Due momenti distinti

```text
PREPARAZIONE OFFLINE (solo Common Voice)                     commonvoice_purescribe.py
  2 extract   audio dev + trascrizioni ufficiali: Whisper autonomo con timestamp per parola
              -> match esatto e monotono con le parole ufficiali -> clip WAV etichettati CV
              -> al massimo N clip per parola; parole con meno di M clip rimosse
  3 train     purescribe train corpus -> nuovo modello .psm                (CLI esistente)
  (le fasi 1 split e 4 validate su Common Voice sono state rimosse: tutto Common Voice è
   training, la valutazione avviene su DeepSpeak)

PIPELINE FINALE (video deepfake, nessun Whisper)                 lost_in_translation.py
  5 transcribe  audio del video -> estrazione traccia (ffmpeg) -> PureScribe -> testo audio
                frame del video -> lip reading della baseline (Auto-AVSR) -> testo video
                [trascrizioni del labiale lette dagli output della baseline, invariate]
  6 compare     testo audio + testo video -> normalizzazione della baseline -> distanza della
                baseline -> soglia 0,5 -> real / fake, per Whisper (baseline) e per PureScribe
```

### Ruolo di Whisper

Whisper (openai-whisper, modello `base`) compare **soltanto nella fase 2**, per proporre
timestamp di parole candidate sul training Common Voice. Riceve solo l'audio:
`transcribe(audio, language="en", word_timestamps=True)`, senza `initial_prompt`, senza
`align()` e senza testo ufficiale. Le sue parole servono solo per il confronto con il testo
ufficiale; l'etichetta di ogni clip è la parola ufficiale Common Voice. Si usa openai-whisper
direttamente e non stable-ts, perché stable-ts rielabora i tempi delle parole (`regroup`,
`gap_padding`, correzioni sul silenzio): i timestamp devono essere quelli restituiti da Whisper.

Nella pipeline finale Whisper non viene eseguito. Le trascrizioni Whisper della baseline sono
**risultati già prodotti** dalla baseline: vengono solo lette, per calcolare con le stesse regole
le metriche di confronto.

### Ruolo di PureScribe

PureScribe è il trascrittore audio della tesi. Viene addestrato (fase 3) con i clip estratti
da Common Voice e poi usato sui video DeepSpeak (fase 5), dove avviene la valutazione.
Riceve sempre e solo l'audio, tramite `Recognizer.transcribe`; il core (HMM-GMM, MFCC,
decoder Viterbi, formato `.psm`) non è stato modificato. Il vocabolario è chiuso: PureScribe
può emettere soltanto le parole addestrate.

### Rapporto con "Lost in Translation" e baseline usata

Non esiste una repository ufficiale del paper (ottobre 2026: `github.com/matyasbohacek` senza
repository pubbliche, `github.com/hfaridlab` con il solo `deepspeak`, nessun link nel PDF).
Il metodo viene dal paper; l'implementazione di riferimento è l'esecuzione locale più recente
della baseline:

- codice: `C:\Users\cofon\DeepSpeak\transcribe.py` e `text_utils.py` (settembre 2026);
- dati: sottoinsieme `test` di DeepSpeak in `D:\DeepSpeak\test\{real,fake}` (letto tramite il
  collegamento `C:\Users\cofon\DeepSpeak\test` → `D:\DeepSpeak\test`); i video usati hanno data
  di modifica 26/06/2026, precedente all'esecuzione;
- output usati: `C:\Users\cofon\DeepSpeak\transcriptions\whisper\test\{real,fake}\*.txt` (ramo
  audio) e `...\avsr\test\{real,fake}\*.txt` (ramo video), scritti il 20/09/2026: 24 video
  autentici e 100 deepfake (20 ciascuno per latentsync, facefusion, hellomeme, liveportrait, memo).

| Elemento | Baseline (Whisper) | Variante della tesi (PureScribe) |
|---|---|---|
| Video | 124 video `test` trascritti dalla baseline | gli stessi file presenti su disco (120) |
| Ramo audio | Whisper-small su tutto l'audio (`transcribe.py`: per default transformers a finestre da 30 s; il motore usato non è registrato negli output) | PureScribe su tutto l'audio |
| Ramo video | Auto-AVSR con landmark da Haar cascade, blocchi di 150 frame (valore predefinito) | **le stesse trascrizioni**, non rieseguite |
| Normalizzazione | `standardize_text`: minuscole, rimozione di `[..]`, `(..)`, um/uh/hmm/mm/mhm/mmm e punteggiatura ASCII, spazi compattati | identica |
| Distanza | `normalized_levenshtein`: Levenshtein / lunghezza massima sui caratteri, 1,0 se solo un testo è vuoto, arrotondata a 4 decimali | identica |
| Decisione | `fake` se distanza > 0,5 | identica |
| Metriche | TP, TN, FP, FN, accuracy, precision, recall, F1, distribuzioni | identiche, sugli stessi video |

`lost_in_translation.py` riproduce queste tre funzioni della baseline. Verifica: le funzioni
riprodotte ritrovano esattamente i 90 testi puliti di `whisper_results.csv` e le 36 distanze e
predizioni di `evaluation_results.csv` salvati dalla baseline.

Differenze note rispetto al paper, lasciate invariate per non cambiare la baseline: il paper
cita la normalizzazione di Yujian e Bo [26], la baseline divide per la lunghezza massima;
il paper usa Whisper "English-only", la baseline Whisper-small multilingue; Auto-AVSR riceve
landmark approssimati da un rilevatore Haar invece del rilevatore previsto.

Una precedente esecuzione di luglio (`D:\DeepSpeak\pipeline_funzionante.py`,
`pipeline_results_sub.csv`) non è usata: limita Whisper ai primi 30 s, conserva la punteggiatura
e 47 dei suoi 100 video non sono più su disco.

## Struttura delle directory

```text
<root del repository>\
  dev_cv_words\<parola>\*.wav   fase 2: clip mono 16 kHz PCM16 (<audio>_w<indice>.wav), 12-23 per parola
  dev_cv_words\clips.jsonl      fase 2: una riga per occorrenza ammessa (provenienza/scarto/potatura)
  dev_cv_words\extraction.json  fase 2: fonti, parametri Whisper, vocabolario finale, conteggi per parola
  dev_cv.psm                    fase 3: modello PureScribe addestrato su dev_cv_words
  cv_words\, purescribe_cv.psm  estrazione (20/10 clip) e modello precedenti, da non sovrascrivere
  deepfake_purescribe.tsv       fase 5: video, stato, trascrizione PureScribe, dettaglio, SHA-256 modello
  deepfake_comparison.json      fase 6: protocollo, metriche Whisper/PureScribe, differenze, per video
```

`corpus\` del repository contiene i dati Common Voice di ingresso (`dev\*.mp3`, `dev.tsv`) e
`words_cv\` (parole singole già estratte dagli audio di `dev`); è protetta e non va usata come
output.

Le fasi 3, 5 e 6 rifiutano output già esistenti; la fase 2 sovrascrive solo una propria
estrazione precedente nella stessa cartella. Dati in sola lettura:
`corpus\`, `D:\DeepSpeak`, `C:\Users\cofon\DeepSpeak`.

## Le fasi

Da PowerShell, nella root del repository. I comandi non usano variabili di sessione: si possono
copiare e incollare così come sono.

### 2. Estrazione delle parole con Whisper

```powershell
.\.venv\Scripts\python.exe commonvoice_purescribe.py extract `
    --data-root corpus `
    --audio-dir corpus\dev `
    --transcription-file corpus\dev.tsv `
    --output dev_cv_words `
    --clips-per-word 23 `
    --min-clips-per-word 12 `
    --whisper-model base
```

- Impostazioni: `--data-root`, `--audio-dir`, `--transcription-file` e `--output` si passano una
  volta; dopo una validazione riuscita dei percorsi vengono salvati (assoluti) in
  `commonvoice_purescribe.settings.json` accanto allo script e riusati quando mancano dalla riga
  di comando. Priorità: riga di comando > percorsi derivati da un `--data-root` esplicito >
  valori ricordati > predefiniti (`corpus` accanto allo script, `<data-root>\dev`,
  `<data-root>\dev.tsv`, `corpus_cv`). Altri predefiniti: `--clips-per-word 23`,
  `--min-clips-per-word 12`, `--whisper-model base`, `--seed 20261014`. Dalla seconda volta basta
  `commonvoice_purescribe.py extract`. Attenzione: il comando sopra salva `dev_cv_words` come
  output ricordato al posto di `cv_words`, quindi un `extract` successivo senza `--output`
  sovrascrive `dev_cv_words`.
- Sovrascrittura: se la cartella di output esiste ed è un'estrazione precedente (`clips.jsonl` o
  `extraction.json` e sottocartelle di soli `.wav`) oppure è vuota, viene svuotata e riscritta,
  dopo la validazione degli input. Ogni altra cartella (per esempio `corpus` del repository, che
  non ha questi file) e ogni cartella che contiene gli input vengono rifiutate senza toccarle.

- Input: `corpus\dev\*.mp3` (default `<data-root>\dev`), `corpus\dev.tsv` per le etichette
  (colonne `path`, `sentence`) e per i metadati (default `<data-root>\<nome cartella audio>.tsv`:
  `path`, `client_id`, `locale`, per le statistiche sui parlanti e per scartare locale diversi da
  `en`), Whisper `base` (cache locale `~\.cache\whisper\base.pt`), ffmpeg nel PATH.
- Nessuno split: ogni audio Common Voice è materiale di training.
- Elaborazione: audio in ordine `random.Random(seed).shuffle` dei nomi ordinati (seed predefinito
  20261014), senza limite sul numero di audio. Un audio viene trascritto solo se il testo
  ufficiale contiene almeno una parola ammessa la cui quota (`--clips-per-word`) non è piena.
  Per ogni parola ufficiale ammessa: match esatto e monotono con le parole Whisper
  (`match_word_indices`), timestamp validato (`valid_word_bounds`), padding 50/120 ms limitato ai
  bordi, durata minima 100 ms, scrittura esclusiva (`save_wav`). A fine estrazione le parole con
  meno di `--min-clips-per-word` clip vengono rimosse (solo i file creati da questa esecuzione).
- Output: `dev_cv_words\<parola>\*.wav`, `dev_cv_words\clips.jsonl`, `dev_cv_words\extraction.json`
  (`purescribe train` legge solo le sottocartelle).
- Costo: circa 0,8 s per audio su CPU nel pilota precedente (134 audio in 104 s); tutti i 16.402
  audio utilizzabili ≈ 3,6 ore. Non rimisurato dopo il refactoring.
- Arresto anticipato: Ctrl+C interrompe il ciclo; i clip già salvati vengono potati con la stessa
  soglia e `extraction.json` viene scritto con `stopped: "interrupted"`.
- Successo: exit code 0; in `extraction.json` `vocabulary` (ogni parola con un numero di clip fra
  `--min-clips-per-word` e `--clips-per-word`) e `pruned_words`. Se restano meno di due parole la
  fase termina con errore (exit code 2) dopo aver scritto `extraction.json`: ripetere senza
  interrompere o con una soglia più bassa.
- Errori prevedibili: cartella di output esistente che non è un'estrazione precedente;
  `--min-clips-per-word` maggiore di
  `--clips-per-word`; ffmpeg assente; modello Whisper non in cache. Errori di lettura o di Whisper
  su un singolo audio non fermano la fase: restano in `clips.jsonl` come `load_error`/`whisper_error`.
  Un arresto brusco (chiusura del terminale, crash) avviene invece prima della potatura e non è
  riprendibile: quel corpus non va addestrato; rilanciare l'estrazione, che lo sovrascrive.

### 3. Addestramento PureScribe

```powershell
.\.venv\Scripts\python.exe -m purescribe.cli train dev_cv_words -o dev_cv.psm
.\.venv\Scripts\python.exe -m purescribe.cli info dev_cv.psm
```

- Input: soltanto `dev_cv_words\` della fase 2 (clip di training).
- Output: `dev_cv.psm`, con le impostazioni predefinite del trainer (2 gaussiane per stato,
  10 iterazioni).
- Successo: exit code 0; `info` elenca come classi le parole del vocabolario presenti nel corpus.
- Errori prevedibili: modello già esistente (la CLI rifiuta di sovrascrivere); corpus vuoto.
- I comandi e le misure delle fasi 5–6 qui sotto si riferiscono ancora al modello precedente
  `purescribe_cv.psm`. Per valutare `dev_cv.psm` si passa `--model dev_cv.psm` con output nuovi
  (per esempio `deepfake_dev_cv.tsv`, `deepfake_comparison_dev_cv.json`): gli output esistenti
  sono rifiutati e non vanno sovrascritti.

### 5. Trascrizione audio dei video con PureScribe

```powershell
.\.venv\Scripts\python.exe lost_in_translation.py transcribe --model purescribe_cv.psm --baseline-transcripts C:\Users\cofon\DeepSpeak\transcriptions --subset test --videos-dir D:\DeepSpeak\test --output deepfake_purescribe.tsv
```

Per continuare un'esecuzione interrotta si rilancia lo stesso comando con `--resume`:

```powershell
.\.venv\Scripts\python.exe lost_in_translation.py transcribe `
    --model purescribe_cv.psm `
    --baseline-transcripts C:\Users\cofon\DeepSpeak\transcriptions `
    --subset test `
    --videos-dir D:\DeepSpeak\test `
    --output deepfake_purescribe.tsv `
    --resume
```

- Input: modello; elenco dei video = nomi dei file della baseline presenti in entrambi i rami
  (`whisper\test\<real|fake>\` e `avsr\test\<real|fake>\`; il contenuto non viene letto);
  video `D:\DeepSpeak\test\<real|fake>\<nome>.mp4`.
- Elaborazione: per ogni video `purescribe.audio.load_audio` estrae l'intera traccia audio con
  ffmpeg (mono, 16 kHz) e `Recognizer.transcribe` la trascrive (stessa decodifica: fino a 30 s in
  un solo passaggio, oltre per segmenti VAD).
- Avanzamento: prima di ogni video `[indice/totale] <video>: extracting audio`, poi la durata
  (`… s of audio, decoding`), infine esito e tempo (`ok in … s: <testo>`); le righe sono scritte
  subito (flush) anche se l'output è rediretto su file.
- Ripresa (`--resume`): il TSV esistente viene accettato solo se l'intestazione è quella della
  fase 5 e ogni riga completa è, nell'ordine, il video successivo della baseline, con lo SHA-256
  del modello indicato e lo stesso `--videos-dir`; le righe complete (anche `missing` ed `error`)
  vengono saltate e non duplicate. Un'ultima riga interrotta (senza terminatore, o con un campo tra
  virgolette non chiuso) viene rimossa con un messaggio e il suo video rielaborato. In ogni altro
  caso la fase si ferma (exit code 2) senza toccare il file. Senza `--resume` un output esistente
  viene rifiutato. Ctrl+C lascia nel file solo righe complete.
- Durata misurata (17/10/2026, CPU, 4 thread per le emissioni): 2,6-2,7 s di calcolo per secondo
  di audio (`real/227-12481.mp4`, 21,96 s: 57,2 s), più circa 50 s per caricare il modello e 8 s
  per preparare il decoder. Stima per i 117 video presenti ancora da trascrivere (1.287,8 s di
  audio): circa un'ora. Memoria di picco misurata: 1,6 GiB di working set (dominata dal modello).
- Output: `deepfake_purescribe.tsv` con stato `ok`, `missing` (video non presente) o `error`.
- Successo: exit code 0; una riga per video della baseline.
- Errori prevedibili: file di output esistente senza `--resume`; TSV non coerente con video,
  ordine, modello o cartella; cartella dei video assente; nessuna trascrizione della baseline per
  il subset; video senza traccia audio o ffmpeg assente (`error` nella riga).

### 6. Confronto con la trascrizione del labiale

```powershell
.\.venv\Scripts\python.exe lost_in_translation.py compare --baseline-transcripts C:\Users\cofon\DeepSpeak\transcriptions --subset test --purescribe deepfake_purescribe.tsv --threshold 0.5 --output deepfake_comparison.json
```

- Input: trascrizioni della baseline (Whisper e lip reading, già normalizzate dalla baseline) e TSV
  della fase 5. Etichetta reale = cartella `real`/`fake`; metodo dal prefisso del nome del file,
  con la regola della baseline.
- Elaborazione: solo i video con trascrizione PureScribe `ok` entrano nel confronto, per entrambi i
  sistemi. La distanza Whisper usa i testi salvati dalla baseline; l'output di PureScribe passa per
  la stessa `standardize_text` che la baseline applicava all'output di Whisper. Stessa distanza e
  stessa soglia per i due sistemi.
- Output: `deepfake_comparison.json` (con SHA-256 dei testi della baseline e del TSV).
- Successo: exit code 0; `videos.evaluated` > 0 e `videos.excluded` coerente con la fase 5.
- Errori prevedibili: output esistente; TSV con video estranei alla baseline, duplicati o prodotto
  da modelli diversi.

**Punto di integrazione.** Le fasi 5–6 leggono il formato di output della baseline
(`transcribe.py --save-txt`): `transcriptions\{whisper,avsr}\<subset>\{real,fake}\<video>.txt`.
Un nuovo dataset trascritto dalla stessa baseline (anche un altro `--subset`) entra senza
modifiche. Il ramo video non viene eseguito da questo repository: Auto-AVSR e il suo ambiente
restano quelli della baseline.

**Ramo labiale alternativo (esperimento separato, opzionale).** Con un export video-only di
`avsr_export.py` (DeepSpeak, vedi `DeepSpeak\docs\avsr_export.md`) i testi labiali vengono dal
nuovo root; i testi Whisper e l'universo dei video restano quelli di `--baseline-transcripts`:

```powershell
.\.venv\Scripts\python.exe lost_in_translation.py compare `
    --baseline-transcripts C:\Users\cofon\DeepSpeak\transcriptions --subset test `
    --purescribe experiments\lit_trials_20261017\D_videos\deepfake_purescribe_ac07f0d8.tsv `
    --avsr-transcripts <root export> --avsr-manifest <root export>\manifest.json `
    --model purescribe_cv.psm --threshold 0.5 --output <nuovo json>
```

- `--avsr-manifest` e `--model` sono obbligatori: lo SHA-256 del modello deve coincidere con
  quello del TSV (il `deepfake_purescribe.tsv` della radice viene da un altro modello ed è
  rifiutato). `--model` è accettato anche nel modo storico, dove controlla soltanto.
- Prima si escludono le righe PureScribe non `ok`; poi ogni video deve avere una voce nel
  manifest (chiavi uniche e interne all'universo). Le voci `empty`/`missing`/`error` sono escluse
  per **entrambi** i sistemi e contate in `videos.excluded_avsr` (con motivo in
  `videos.excluded_avsr_videos`). Per `ok` si leggono i byte del TXT e se ne verifica lo SHA-256
  prima di decodificarli: TXT assente o hash diverso → errore, nessun output.
- Il JSON aggiunge root, manifest e suo hash, checkpoint, preprocessing e percorso del modello.
  Senza i nuovi flag il JSON è identico byte per byte a quello precedente.
- Ogni run confronta Whisper e PureScribe sullo stesso testo labiale e sugli stessi video; le
  metriche di run diversi (baseline del 20/09/2026 contro nuovo export) non vanno mescolate, perché
  cambiano anche fps, coda e video esclusi. Run del 2026-10-19:
  `experiments\avsr_compare_20261019\comparison_ac07f0d8_avsr_export_20261019.json`.

## Metriche prodotte

Fase 2 (`extraction.json`): audio trascritti e saltati (per motivo), esiti per audio, clip
estratti e tenuti, vocabolario finale, parole rimosse con il loro numero di clip, esiti su tutte
le parole e, per ogni parola incontrata (tenuta, potata o mai salvata), audio tentati, occorrenze
considerate, match esatti, clip accettati, parlanti disponibili e usati, motivi di scarto.

`score_predictions` (WER, S/D/I, frasi esatte, IV/OOV) resta disponibile in
`commonvoice_purescribe.py`, ma nessun comando della pipeline lo usa più dopo la rimozione della
fase 4.

### Diagnosi indipendente su Common Voice test (fuori dalla pipeline)

```powershell
.\.venv\Scripts\python.exe -B experiments\quality_baseline_20261019\validate.py --model purescribe_cv.psm --phase5-tsv experiments\lit_trials_20261017\D_videos\deepfake_purescribe_ac07f0d8.tsv --output <nuovo file>.json
```

Rivaluta con `score_predictions` le ipotesi già decodificate sui set congelati `eval_random` ed
`eval_covered` (split ufficiale `test`, parlanti disgiunti dal training) dopo averne verificato gli
SHA-256 (`experiments\quality_baseline_20261019\manifest.json`); con `--phase5-tsv` rifiuta un TSV
le cui righe non portano lo SHA-256 del `--model`. È una verifica di generalizzazione separata: non
decodifica audio e non modifica le fasi 2, 3, 5, 6 né la soglia. Risultati in
`experiments\quality_baseline_20261019\REPORT.md`.

Fase 6 (`deepfake_comparison.json`), per `whisper_baseline` e per `purescribe`: TP, TN, FP, FN
(classe positiva `fake`), accuracy, precision, recall, F1, distribuzione delle distanze (n, min,
mediana, media, max) per autentici e deepfake e per metodo di generazione, accuratezza per
metodo; differenza assoluta e relativa PureScribe − Whisper; distanze e predizioni per video.
Fra i metodi del sottoinsieme, latentsync è lip-sync; facefusion è face-swap; hellomeme,
liveportrait e memo sono avatar animati. Con 100 deepfake e 20 autentici, un classificatore che
dichiara tutto `fake` ottiene accuracy 83,3% e F1 90,9%: leggere sempre TN e FP.

## Limitazioni

- **Vocabolario chiuso.** PureScribe trascrive solo le parole addestrate; nel parlato libero la
  maggior parte delle parole è fuori vocabolario e la distanza dal labiale resta alta anche per i
  video autentici.
- **Dominio acustico diverso.** Clip MP3 Common Voice, tagliate da frasi lette, contro audio
  Opus di webcam DeepSpeak.
- **Confini delle parole.** Il match testuale rende un timestamp un candidato, non prova che il
  confine acustico sia corretto. Non è stato fatto ascolto sistematico dei clip.
- **Dati della baseline.** 4 dei 24 video autentici trascritti dalla baseline non sono più su disco;
  il confronto riguarda 120 video (20 autentici, 100 deepfake), per entrambi i sistemi. Un solo
  metodo lip-sync (latentsync); diff2lip non fa parte di questo sottoinsieme. Classi sbilanciate.
- **Baseline diversa dal paper** (vedi sopra) e clip DeepSpeak brevi (mediana 6,6 s contro i
  30–300 s del paper). L'identità dei video è verificata da percorso e data di modifica; la
  baseline non registrava hash dei file.
- **Soglia non ritarata.** La soglia 0,5 resta quella della baseline anche per PureScribe, per
  non attribuire alla soglia un cambiamento dovuto al trascrittore.

## Dove si trova il codice

- `commonvoice_purescribe.py`: fase 2 e le funzioni che usa (`read_reference_map`,
  `read_metadata`, `official_words`, `_comparison_word`, `match_word_indices`,
  `valid_word_bounds`, `save_wav`), più `score_predictions`.
- `purescribe/`: il riconoscitore e la CLI della fase 3. Il 17/10/2026 è cambiata solo
  l'implementazione della ricerca in `purescribe/decoder.py`, senza cambiare il risultato: con il
  lessico uniforme il miglior predecessore di ogni parola è la migliore uscita (nessuna matrice
  V x V), la ricerca è vettorializzata sugli stati e le emissioni sono calcolate frame per frame
  solo per gli stati che possono essere vivi. Modello, formato `.psm`, vocabolario, beam, VAD e
  feature sono invariati; il caso con bigrammi usa ancora la matrice completa.
- `lost_in_translation.py`: fasi 5 e 6, con le funzioni del protocollo della baseline
  (`standardize_text`, `normalized_levenshtein`, `method_of`). Non importa Whisper né gli altri
  script.

## Test

I test in `tests/` verificano con audio sintetico, in directory temporanee, matching, timestamp,
scrittura dei WAV, lettura delle trascrizioni, estrazione con quote e potatura (Whisper simulato), WER, esecuzione delle CLI `train` e `transcribe` e
protezione del modello in `train`. `tests/test_decoder_equivalence.py` confronta bit per bit
(parole, segmenti, punteggio) il decoder con la sua versione precedente, congelata in
`tests/reference_decoder.py`, su modelli sintetici con lessico uniforme e bigrammi, silenzio,
pareggi, beam e `max_active`; `tests/test_transcribe_resume.py` verifica avanzamento e
`--resume` della fase 5 con un riconoscitore simulato:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp $env:TEMP\purescribe_pytest
```

Le verifiche svolte per questa pipeline non sono test permanenti: sono documentate in
`verify/thesis-lit-purescribe/REVIEW-LOG.md`.
