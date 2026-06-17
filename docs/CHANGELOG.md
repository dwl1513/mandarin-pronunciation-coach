# Changelog

All notable changes to the Mandarin Pronunciation Coach are recorded here.
This file is the canonical place to look up *why* the algorithm differs from
the original design document.

## v0.1.0 — initial submission

### Added
- Full M1-M5 pipeline (preprocess → ASR/align → features → 5-dim scoring → report)
- Gradio web UI with waveform / spectrogram / F0-comparison / score-bar tabs
- edge-tts default reference voice, Aliyun Qwen-TTS optional reference voice,
  Xiaomi MiMo-V2.5-TTS optional reference voice, multi-reference scoring,
  and F5-TTS path kept as optional fallback
- 56 unit + integration tests
- SpeechOcean762 PCC evaluation skeleton (`src/evaluation/speechocean_eval.py`)
- Error-simulation benchmark script (`scripts/benchmark_pronunciation.py`) for showing dimension-specific score responses
- Initial/final sub-syllable accuracy scoring based on pinyin boundary
  estimation and MFCC-DTW, surfaced in the per-character report
- Articulation coverage cue for accuracy scoring, comparing voiced-frame
  coverage and local duration with the TTS reference window
- Per-character completeness localization based on ASR/reference LCS matching,
  surfaced as 已读 / 漏读 in the report
- Score confidence estimation based on valid speech duration, TTS reference
  availability, ASR coverage, F0 availability, and articulation coverage
- Local benchmark variants: dropped-middle syllable, muted-middle syllable,
  and local pitch shift for showing word-level pronunciation errors

### Fixed during development (visible in commit history)

#### Bug 9 — missing or muted local speech was hard to locate
The first benchmark only had a sentence-level completeness score.  When the
middle of an utterance was muted or deleted, the system could lower overall
completeness, but the per-character report did not say which reference
character was missing.  **Fix**: add an LCS-based character coverage pass
between ASR output and reference text.  Each reference character is now marked
as 已读 or 漏读 in the report.

#### Bug 11 — high/low scores lacked reliability context
The report used to show only scores.  A short, noisy, unvoiced, or ASR-failed
recording could still produce numeric values, but the user had no way to know
how much evidence supported those values.  **Fix**: add a confidence scorer
that combines effective speech duration, reference availability, ASR coverage,
F0 availability, and articulation coverage.  The report now shows overall
评分可信度 and per-character 高 / 中 / 低 labels.

#### Bug 10 — MFCC-DTW alone could over-score swallowed syllables
When a local segment was missing, forced alignment sometimes still found a
nearby acoustically similar window.  A pure MFCC-DTW score could therefore
stay high.  **Fix**: fuse MFCC-DTW with an articulation cue from voiced-frame
coverage and local duration.  Low voiced coverage now adds “有效发声不足” to
the per-character note.

#### Bug 8 — per-character accuracy was too coarse for feedback
The original accuracy score only gave one MFCC-DTW number per character.  This
could flag that a character sounded different from the reference, but it could
not say whether the difference was closer to the 声母 or the 韵母.  **Fix**:
estimate a pinyin-based initial/final boundary inside each aligned character
window and run MFCC-DTW on the two sub-segments separately.  The report now
shows 声母分 and 韵母分, and low sub-scores are turned into 声母偏差 /
韵母偏差 notes.

#### Bug 1 — wav2vec2 loading on torch < 2.6
`transformers` enforces CVE-2025-32434 by refusing `torch.load` on `.bin`
checkpoints unless torch ≥ 2.6.  **Fix**: pass `use_safetensors=True` to
`Wav2Vec2ForCTC.from_pretrained` so the loader uses HuggingFace's
auto-converted safetensors variant when needed.

#### Bug 5 — dictionary tones caused false tone errors
`pypinyin` returns dictionary tones, but connected Mandarin uses common tone
sandhi such as 三声连读, 一/不 变调.  This made correct readings like "你好" and
"不是" show up as per-character tone mistakes.  **Fix**: apply common
Mandarin tone-sandhi rules while parsing the reference text, and surface both
dictionary tone and scoring tone in the report.

#### Bug 6 — rule-only tone scoring penalized high-quality TTS
Aliyun Qwen-TTS used as both user audio and reference audio should score near
perfect on tone, but the v2 classifier gave only **58.97 / 100** because it
forced connected-speech F0 contours into isolated tone labels.  **Fix**:
add reference-driven per-character F0 scoring.  User and reference F0 curves
are converted to semitones, centered by utterance baseline, compared with DTW,
and fused with slope and voiced-coverage scores.  On "今天天气真好。", the same
Qwen-TTS self-check now scores **98.96 / 100** on tone.

#### Bug 7 — fluency over-penalized formal TTS readings
Formal standard readings from Qwen-TTS can be slower and include demonstration-style
pauses.  The old fluency formula treated that as disfluency and gave only
**52.12 / 100** on a self-check.  **Fix**: use a wider natural reading-rate
band, allow short-sentence demonstration pauses, and add rhythm stability from
per-character duration variance.  The same Qwen-TTS self-check now scores
**82.48 / 100** on fluency, while MiMo self-check remains near perfect.

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
- Complex tone sandhi — common rules like 三声连读 and 一/不 变调 are handled,
  but phrase-level sandhi is broader than the current rule set.
- Sentence-final lowering compresses tone 3's dipping contour on the last
  syllable, causing systematic tone 3 → tone 1 confusion at sentence ends.
- 儿化 (rhotacization) — currently the 儿 is treated as a separate character
  with its own alignment window; in fluent speech it's a vowel offglide of
  the preceding syllable.
