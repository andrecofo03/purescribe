# Common Voice: costruzione del corpus PureScribe (fase 2)

Comandi, input e output delle fasi sono in [pipeline.md](pipeline.md). Qui sono descritti
il contratto sui dati e le regole del corpus implementati in `commonvoice_purescribe.py`.

## Sorgenti (lettura verificata il 17 ottobre 2026, senza Whisper)

I dati Common Voice stanno nella cartella `corpus\` del repository (`--data-root` predefinito).

| Sorgente | Formato | Uso |
|---|---|---|
| `corpus\dev` | 16.402 MP3 | Bacino audio predefinito (`--audio-dir`); nessun fallback verso altre cartelle |
| `corpus\dev.tsv` | UTF-8, separatore TAB con intestazione, campi senza quoting | Unica fonte delle etichette (colonne `path`, `sentence`; `--transcription-file`) e dei metadati (`client_id`, `locale`; `--metadata-file`) |
| `corpus\words_cv` | `<parola>\*.wav` | Parole singole già estratte dagli audio di `dev`; non è letta né scritta da `extract` |

`read_reference_map` riconosce il TSV di Common Voice dall'intestazione (più di due colonne, fra
cui `path` e `sentence`) e lo legge con `csv.reader(..., delimiter="\t", quoting=csv.QUOTE_NONE,
strict=True)`, prendendo i campi per nome di colonna: le virgolette nel testo restano letterali.
Un file esplicito a due campi (`<file.mp3>\t<frase>`, quoting CSV, intestazione facoltativa
`path\tsentence`) resta accettato. I nomi sono associati ai file locali senza distinzione tra
maiuscole e minuscole (come Windows); righe duplicate, percorsi non sicuri e righe selezionate
malformate (numero di campi diverso dall'intestazione) fermano la fase.

Lettura reale di `dev.tsv`: 16.403 record, 1 relativo a un audio non presente in `dev`
(`common_voice_en_18679937.mp3`); tutti i 16.402 audio di `dev` hanno una frase (locale `en`).
L'esportazione precedente `cv_path_transcription.txt` (1.048.576 record dell'intero Common Voice)
coincide con `dev.tsv` sui 16.402 audio a meno delle virgolette, che il confronto delle parole
ignora, tranne un record corrotto da una virgoletta non chiusa (`common_voice_en_18679934.mp3`,
che vi risultava su più righe e veniva scartato): non è più necessaria.

## Nessuno split

Tutti i clip estratti da Common Voice servono solo all'addestramento del modello acustico: il
sistema viene valutato a valle su un altro dataset (DeepSpeak, `D:\DeepSpeak\test`). Non c'è
quindi una partizione train/validation di Common Voice, né deduplicazione per hash del contenuto
MP3. Gli audio vengono considerati nell'ordine `random.Random(seed).shuffle` dei nomi ordinati
(seed predefinito **20261014**, registrato in `extraction.json`), per non concentrare i clip
di una parola sulle registrazioni consecutive dello stesso parlante.

## Vocabolario dinamico

Il vocabolario non è fissato in anticipo: nasce dall'estrazione.

- Parola ammessa: composta da lettere, con eventuali `'` o `-` fra lettere. Così `we're` e
  `re-sign` sono classi distinte da `were` e `resign`, e ogni etichetta è un nome di cartella
  valido su Windows (esclusi `con`, `prn`, `aux`, `nul`). Token con cifre sono ignorati.
- Quota massima (`--clips-per-word`, predefinita 23): raggiunta la quota, le occorrenze
  successive della parola diventano `quota_reached`; così le parole molto frequenti ("the",
  "and") non dominano il corpus.
- Soglia minima (`--min-clips-per-word`, predefinita 12): a fine estrazione le parole con meno
  clip vengono rimosse dal disco, perché con 1–2 campioni per stato le covarianze diagonali delle
  gaussiane collassano. Il corpus finale contiene solo parole con un numero di clip compreso fra
  la soglia e la quota.

## Estrazione

1. Un audio è trascritto solo se il suo testo ufficiale contiene una parola ammessa con quota non
   piena: il testo serve a scegliere i candidati, non viene mai passato a Whisper. Non c'è un
   limite al numero di audio: si arriva alla fine di `dev` oppure a un Ctrl+C, dopo il quale la
   potatura e il riepilogo vengono eseguiti comunque.
2. Whisper (`--whisper-model`, predefinito `base`) trascrive l'audio da solo:
   `model.transcribe(audio, language="en", word_timestamps=True)`; nessun `initial_prompt`,
   nessun `align()`.
3. Le parole ufficiali (frase divisa per spazi) e quelle di Whisper sono confrontate con
   `match_word_indices`: blocchi uguali di `difflib.SequenceMatcher(autojunk=False)` sulle
   forme normalizzate, quindi match esatti, monotoni, mai riusati e senza fuzzy matching.
4. Normalizzazione (`_comparison_word`): `casefold()`, `’`/`‘` → `'`, rimozione della sola
   punteggiatura esterna. `we’re` = `we're`; `we're` ≠ `were`; `re-sign` ≠ `resign`.
   Riferimento `the quick brown fox` e Whisper `the quick fox`: si salvano `the`, `quick`, `fox`.
5. Per ogni occorrenza ufficiale di una parola ammessa:
   - nessun match → `unmatched` (inserzioni, omissioni e sostituzioni sono ignorate);
   - start/end mancanti, non numerici, non finiti, negativi, invertiti, nulli o oltre la
     durata → `invalid_timestamp` (`valid_word_bounds`); nessuna correzione;
   - quota della parola piena → `quota_reached`;
   - padding 50 ms prima e 120 ms dopo, limitato ai bordi dell'audio; clip sotto i 100 ms →
     `short_clip`;
   - altrimenti `saved`: WAV mono, 16 kHz, PCM signed 16 bit in
     `<output>\<parola ufficiale>\<audio>_w<indice>.wav`, creato in modo esclusivo
     (`save_wav`, che in caso di errore rimuove solo il file appena creato e rilancia
     l'errore originale).
6. Potatura: le parole sotto la soglia minima perdono i clip salvati da questa esecuzione e la
   loro cartella. Se la cartella di output contiene un'estrazione precedente, questa viene
   svuotata prima di iniziare; una cartella con altri contenuti viene rifiutata senza toccarla.
7. Ogni clip proviene da un'occorrenza reale distinta: nessun file viene duplicato per riempire
   una classe.

### Provenienza

`clips.jsonl` (nella directory di output) contiene una riga per ogni occorrenza di parola ammessa
negli audio trascritti: `audio` sorgente, `client_id` (se `dev.tsv` è disponibile), `word_index`
(posizione nella frase ufficiale divisa per spazi, esclusi i token di sola punteggiatura),
`official_word` (token originale), `label`, `status`, e per i clip salvati
`whisper_start`/`whisper_end` (tempi Whisper usati), `clip_start`/`clip_end` e `wav`. Le righe
`load_error`/`whisper_error` riportano il messaggio di errore. In coda, una riga `pruned` per ogni
clip rimosso dalla potatura.

`extraction.json` registra sorgenti, ordine e seed, parametri (modello e chiamata Whisper, quota,
soglia, limite di audio, formato dei clip), motivo dell'arresto, audio saltati (`quota_full` o
`no_eligible_word`), vocabolario finale, parole rimosse e conteggi per **ogni** parola incontrata
negli audio trascritti (tenuta, potata o mai salvata, campo `kept`): audio candidati trascritti,
occorrenze considerate, match esatti, clip accettati, parlanti disponibili e usati (solo
`client_id` noti, se `dev.tsv` è disponibile), motivi di scarto (compreso `quota_reached`) e
occorrenze negli audio saltati senza trascrizione perché tutte le loro parole erano a quota.

## Cosa non è dimostrato

Un match testuale rende il timestamp un candidato, non prova che il confine acustico sia
corretto. Nessun clip è stato ascoltato sistematicamente; non si afferma correttezza fonetica.
