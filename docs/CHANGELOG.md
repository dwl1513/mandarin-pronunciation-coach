# Changelog

All notable changes to the Mandarin Pronunciation Coach are recorded here.
This file is the canonical place to look up *why* the algorithm differs from
the original design document.

## v0.1.0 — initial submission

### Added
- Full M1-M5 pipeline (preprocess → ASR/align → features → 5-dim scoring → report)
- Gradio web UI with waveform / spectrogram / F0-comparison / score-bar tabs
- edge-tts default reference voice (F5-TTS path kept as optional fallback)
- 36 unit + integration tests; ~3.5 min on CPU
- SpeechOcean762 PCC evaluation skeleton (`src/evaluation/speechocean_eval.py`)

### Fixed during development (visible in commit history)

#### Bug 1 — wav2vec2 loading on torch < 2.6
`transformers` enforces CVE-2025-32434 by refusing `torch.load` on `.bin`
checkpoints unless torch ≥ 2.6.  The host conda env ships torch 2.5.1 to
stay compatible with F5-TTS.  **Fix**: pass `use_safetensors=True` to
`Wav2Vec2ForCTC.from_pretrained` so the loader uses HuggingFace's
auto-converted safetensors variant.

#### Bug 2 — CTC peak-alignment starves tone scoring
`torchaudio.functional.forced_align` returns a path where each character
"emits" for only 1–3 frames (20–60 ms) and the rest is blank.  Slicing F0
by these raw spans gave **most syllables fewer than 3 voiced frames** and
the tone classifier returned "undetectable" for everything.

**v1 attempt — midpoint expansion**: extend each span to share the blank
gap with its neighbours via a 50/50 midpoint split.  This **over-corrected**
— when neighbouring syllables were far apart the expanded window bled into
the next syllable's vowel, systematically inverting tone 2 (rising) into a
falling shape.

**v2 — capped midpoint expansion**: extend the span outward by at most 4
model frames (≈80 ms) per side, never past the neighbour's midpoint.  This
covers the typical Mandarin vowel half-length without bleed.  See
`src/asr/aligner.py:_spans_from_alignment`.

#### Bug 3 — per-syllable z-score collapses tone 1 / tone 3
The original tone classifier (per design doc) z-scored each syllable's F0
before template matching.  This kills the absolute pitch position that
distinguishes **tone 1 (high level)** from **tone 3 (low level)** — both
look like a flat zero-mean contour after normalisation.  On TTS audio the
classifier missed 4 out of 6 syllables in "今天天气真好。", crashing the
tone dimension to 24.5 / 100.

**Fix**: rewrite the classifier to use **utterance-level F0 baseline**
(median + IQR over all voiced frames in the clip) plus four interpretable
features per syllable — relative position, linear slope, quadratic
curvature, range.  Decision tree ordered tone 3 → 4 → 2 → 1 → 5.  Tone
score jumped 24.5 → 69.9 on the same clip without touching any other
module.  See `src/scoring/tone.py:_classify_by_features`.

#### Bug 4 — ASR vocabulary leaks pinyin / English tokens
The `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` vocab contains a
handful of non-CJK tokens (pinyin trigrams, ASCII letters).  On
out-of-distribution audio the model occasionally emits these between
correct Chinese characters, e.g. `'今 TI天 B步 B吧'`.  **Fix**: strip
everything that isn't a CJK codepoint or whitespace from the decoded
output.  See `src/asr/recognizer.py:recognize`.

### Known limitations (deferred)
- Tone sandhi (3+3 → 2+3, 一/不 变调) — pypinyin doesn't apply context-aware
  sandhi; expected tones in the report reflect dictionary form, not actual
  sentence pronunciation.
- Sentence-final lowering compresses tone 3's dipping contour on the last
  syllable, causing systematic tone 3 → tone 1 confusion at sentence ends.
- 儿化 (rhotacization) — currently the 儿 is treated as a separate character
  with its own alignment window; in fluent speech it's a vowel offglide of
  the preceding syllable.
