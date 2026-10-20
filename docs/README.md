# PureScribe

Speech recognition built from first principles in numpy and scipy.

No neural networks, no language models in the modern sense, no pretrained
weights, no network calls. Audio goes in, a signal chain you can read end to
end turns it into text:

```
waveform -> pre-emphasis -> framing/windowing -> |FFT|^2 -> mel filterbank
         -> log -> DCT -> MFCC + delta + delta-delta
         -> Gaussian-mixture HMM per word
         -> token-passing Viterbi search over a word loop
         -> text with timings
```

This is how speech recognition worked between roughly 1990 and 2012, and on a
**constrained vocabulary** it still works well. On the bundled ten-digit
benchmark it scores 100% isolated-word accuracy and 0.0% word error on
connected digit strings.

It is not a dictation system. Open-vocabulary transcription of meetings needs a
large-vocabulary acoustic model trained on thousands of hours; that is a
different machine and this one will not pretend to be it.

## Install

```bash
pip install -e .
```

Requires Python 3.9+, numpy and scipy. `ffmpeg` is optional and only needed for
non-WAV input.

## Quick start

No corpus handy? Generate one with the included formant synthesiser, which
builds spoken digits from a source-filter model rather than recordings:

```bash
python examples/make_corpus.py corpus --per-word 24
purescribe train corpus/train -o digits.psm
purescribe transcribe corpus/sequences/seq_000.wav -m digits.psm
# eight six two
python examples/evaluate.py digits.psm corpus
# isolated-word accuracy: 60/60 = 100.0%
# connected-word WER:     0.0%  (sub 0, del 0, ins 0 over 163 words)
```

## Training on your own audio

Record clips and sort them into one directory per word:

```
corpus/
  yes/    clip001.wav clip002.wav ...
  no/     ...
  cancel/ ...
```

Then:

```bash
purescribe train corpus -o commands.psm --mixtures 2
purescribe transcribe recording.wav -m commands.psm -f timed
```

Roughly 20+ clips per word, spread over every speaker and microphone you expect
to meet, is where accuracy becomes reliable. Below about 12 clips per word the
models start to overfit: a 39-dimensional diagonal Gaussian has 78 free
parameters, and a state that owns fewer frames than that cannot estimate them.

If you have connected-speech recordings with transcripts, a second training
stage re-aligns whole utterances and refines the word-boundary states:

```python
from purescribe.train import embedded_reestimate, load_manifest
items = load_manifest("utterances.jsonl", rec.feature_config)   # {"audio":..., "text":...}
embedded_reestimate(rec, items, n_iter=4)
```

## Python API

```python
from purescribe import Recognizer

rec = Recognizer.load("digits.psm")
t = rec.transcribe("recording.wav")
print(t.text)
for w in t.words:
    print(f"{w.start:.2f}-{w.end:.2f}  {w.text}")
print(t.to_srt())
```

## Commands

| Command | Purpose |
|---|---|
| `purescribe train <corpus>` | fit word models from `<corpus>/<word>/*.wav` |
| `purescribe transcribe <audio>` | decode to `text`, `timed`, `json` or `srt` |
| `purescribe classify <audio>` | single-word classification of a whole clip |
| `purescribe info <model>` | topology, vocabulary and parameter counts |
| `purescribe features <audio>` | inspect the front end and endpointing |

## How it works

**Front end** (`dsp.py`, `features.py`). Speech is only stationary over 20-30
ms, so it is cut into 25 ms frames every 10 ms. Pre-emphasis flattens the
roughly -6 dB/octave source tilt. A Hamming window keeps sidelobes near -43 dB
instead of the -13 dB a rectangular cut would give. Mel-spaced triangular
filters impose cochlear frequency resolution and smooth away pitch harmonics,
since the phoneme lives in the spectral envelope and not in F0. The DCT then
decorrelates the log filterbank energies, which is what makes the diagonal
covariances downstream defensible. Deltas and delta-deltas smuggle local
trajectory into each frame, partially repairing the HMM's false assumption that
frames are conditionally independent.

**Acoustic model** (`gmm.py`, `hmm.py`). One left-to-right HMM per word. States
advance or self-loop but never go back, so the topology encodes that an
utterance has direction and the self-loop probability models duration as a
geometric distribution. Each state emits from a diagonal Gaussian mixture.
Training is a flat start followed by segmental k-means (`viterbi`) or soft EM
(`baum_welch`); both are guaranteed non-decreasing in likelihood and are tested
for it.

**Search** (`decoder.py`). Token passing: one Viterbi pass over every word model
at once, with models wired into a loop so a token leaving one may enter any
other, paying a bigram score at the boundary. Word-link records turn the
surviving path back into a transcript with timings. Because the search is
global, word boundaries are decided by the whole utterance rather than greedily.

## Configuration worth knowing

**`lm_scale` and `word_penalty`.** Acoustic likelihoods over 39 correlated
dimensions are hugely over-confident, so the language model must be scaled to
have any effect. The two interact: a uniform LM over V words contributes a
constant `lm_scale*log(1/(V+1))` at every word entry. Keep the net entry cost
negative. If it goes positive the search is *paid* to cross word boundaries and
will emit silence forever to collect the bonus.

**`cmvn_mode`** defaults to `global`: one mean and variance estimated over the
training corpus, stored in the model, applied identically at training and
decoding. Per-utterance normalisation (`utterance`) is the textbook choice for
removing a channel, but it is only valid when the normalisation window holds a
representative mix of phones. On a single-word clip the cepstral mean largely
*is* the word identity, and subtracting it destroys exactly what the model
needs. That is a real measured failure here, not a hypothetical.

**`--fmax`.** The filterbank runs to Nyquist by default. Low-bitrate codecs
discard the top of the band outright - a 24 kbps MP3 measured here was 68 dB
down above 6 kHz - and a model that relies on that band fails completely on such
audio. Training with `--fmax 4000` cut word error on that codec from 100% to
13% with no loss on clean audio. Train on the channel you intend to decode, and
band-limit when you cannot.

## Known limitations

- **Closed vocabulary.** Every word must have a model. There is no way to
  recognise a word the system was not trained on, and out-of-vocabulary speech
  is forced onto whatever model fits least badly.
- **Channel sensitivity.** Models learn the microphone and room they were
  trained in. A different channel degrades accuracy sharply; see `--fmax`.
- **Whole-word models.** There are no phonemes and no pronunciation lexicon, so
  a new word needs new recordings rather than a dictionary entry. This is the
  right trade below a few hundred words and the wrong one above it.
- **No speaker adaptation, no diarisation, no punctuation.**

## Tests

```bash
pytest tests/
```

66 tests. Where a property is guaranteed by the mathematics it is asserted
directly - EM never decreasing likelihood, forward score dominating Viterbi,
forward and backward agreeing on total likelihood, delta of a ramp being its
slope, pre-emphasis matching `|1 - a*exp(-jw)|`. A regression test that only
compares against a previous run of the same code cannot tell you the DSP was
ever right.

## Layout

| Path | Contents |
|---|---|
| `purescribe/audio.py` | decoding, resampling, mono conversion |
| `purescribe/dsp.py` | pre-emphasis, framing, windowing, spectra |
| `purescribe/features.py` | mel filterbank, MFCC, deltas, normalisation |
| `purescribe/vad.py` | energy/ZCR endpointing with hysteresis |
| `purescribe/gmm.py` | diagonal Gaussian mixtures, EM, k-means++ |
| `purescribe/hmm.py` | left-to-right HMM, forward/backward, Viterbi |
| `purescribe/lexicon.py` | vocabulary and counted bigram |
| `purescribe/decoder.py` | token-passing connected-word search |
| `purescribe/train.py` | bootstrap and embedded re-estimation |
| `purescribe/recognizer.py` | the assembled system, save/load |
| `examples/synth.py` | formant synthesiser for the demo corpus |
| `examples/evaluate.py` | isolated accuracy and word error rate |
