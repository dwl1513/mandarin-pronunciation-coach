# 🗣️ Mandarin Pronunciation Coach · 普通话 AI 发音教练

[![python](https://img.shields.io/badge/python-3.10-blue.svg)]()
[![tests](https://img.shields.io/badge/tests-56%20passing-brightgreen.svg)]()
[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()
A Mandarin pronunciation assessment + correction system.  The user
reads (or speaks freely) a given text into the microphone; the system scores
their pronunciation along five dimensions and gives **per-character** feedback
plus a TTS-generated standard reference.

## ✨ Features

- **5-dim scoring** aligned with the official 普通话水平测试 (PSC) rubric
  - 声韵母准确度 (DTW on MFCC vs. TTS reference)
  - **声调** — utterance-level F0 baseline + feature classifier + reference-driven F0 contour matching (project differentiator)
  - 流利度 (语速 / 停顿 / 停顿占比)
  - 韵律自然度 (sentence-level F0 contour DTW + pitch range)
  - 完整度 (CER against ASR transcription)
- **Per-character diagnosis** with tone-error highlighting plus initial/final
  sub-syllable accuracy scores
- **Standard pronunciation playback** via Microsoft edge-tts, Aliyun Qwen-TTS,
  Xiaomi MiMo-V2.5-TTS, or optional F5-TTS
- **Cloud ASR option** via Aliyun Qwen-ASR for more stable completeness scoring
- **Multi-reference scoring** with MiMo + Qwen reference voices to reduce single-TTS style bias
- **Confidence estimation** for score reliability based on ASR coverage, F0
  availability, voiced coverage, speech duration, and reference availability
- **Live waveform / spectrogram / F0 contour visualizations**
- **Gradio web UI** for plug-and-play demo
- Hand-rolled DSP modules (pre-emphasis, framing, MFCC, F0, VAD) — satisfies the "必须有算法实现" course requirement

## 🏗️ Architecture

```
                Reference text ──► pypinyin ──► (char, pinyin, tone)
                                                       │
   Mic / file ──► M1 preprocess ──► M3 features ───────┤
                       │                  │            │
                       └─► VAD ───────────┤            │
                                          │            ▼
                                  M2 wav2vec2 forced alignment
                                          │            │
                                          ▼            ▼
                       ┌────────────────────────────────────────┐
                       │   M4 scoring (accuracy / tone /        │
                       │       fluency / prosody / completeness)│
                       └────────────────────────────────────────┘
                                          │
                                          ▼
                  M4.5 aggregate ──► M5 report + visualize + TTS示范
```

| Module | Responsibility | "From-scratch" vs "uses pretrained" |
|---|---|---|
| `src/audio/preprocess.py` | Pre-emphasis, framing, windowing, VAD | **From scratch (DSP)** |
| `src/features/spectral.py` | MFCC / FBank / energy / ZCR | Wraps librosa primitives |
| `src/features/pitch.py` | F0 via pYIN, semitone normalization | Wraps librosa.pyin |
| `src/asr/recognizer.py` | ASR for completeness | Pretrained wav2vec2-CTC / Aliyun Qwen-ASR |
| `src/asr/aligner.py` | Forced alignment + VAD fallback | torchaudio.functional.forced_align |
| `src/scoring/accuracy.py` | Per-syllable + initial/final DTW MFCC matching | **From scratch (algorithm)** |
| `src/scoring/tone.py` | F0 feature-based tone classifier + sandhi handling | **From scratch (algorithm)** |
| `src/scoring/fluency.py` | Speech rate / pause statistics | **From scratch (algorithm)** |
| `src/scoring/prosody.py` | Contour DTW + pitch range | **From scratch (algorithm)** |
| `src/scoring/completeness.py` | Character-level CER | jiwer |
| `src/feedback/tts.py` | Reference synthesis | edge-tts / Aliyun Qwen-TTS / MiMo-V2.5-TTS / F5-TTS |
| `src/app/gradio_app.py` | Web UI | Gradio |

## 🚀 Quick start

### 1. Environment

This project uses `uv` for dependency and virtualenv management.  The
project Python version is pinned to 3.10 by `.python-version`.

```bash
uv venv --python 3.10
uv sync
```

`requirements.txt` is kept for reference, but `pyproject.toml` and `uv.lock`
are the canonical project environment files.

### 2. Run the unit tests (≈3.5 min, no network / no model download)

```bash
uv run python -m pytest -q
```

You should see `56 passed`.

### 3. Launch the Gradio app

```bash
uv run python -m src.app.gradio_app
```

The first time you press "开始评测" the system will:

1. download the wav2vec2 Chinese ASR/alignment checkpoint (~1.2 GB) — cached
   under `data/cache/hf/`;
2. ping the selected TTS engine to synthesize the reference audio — cached under
   `data/standard_audio/`.

Subsequent runs are fast and fully offline if the selected reference audio is
already cached.

### 4. CLI smoke check (optional)

```bash
uv run python scripts/smoke_check.py
```

To use Aliyun Qwen-TTS as the reference voice:

```bash
uv sync --extra aliyun
export DASHSCOPE_API_KEY=your_key
export TTS_ENGINE=aliyun-tts
uv run python scripts/smoke_check.py --tts-engine aliyun-tts --tts-voice Neil
```

To use Xiaomi MiMo-V2.5-TTS as the reference voice:

```bash
uv sync --extra mimo
export MIMO_API_KEY=your_key
export TTS_ENGINE=mimo-tts
uv run python scripts/smoke_check.py --tts-engine mimo-tts,aliyun-tts --asr-engine aliyun-asr
```

This runs TTS → preprocess → align → all 5 scorers against a built-in
sentence and prints the result.  Use it to verify the live model + network
path on a new machine.

To generate an error-simulation benchmark table for reports / slides:

```bash
uv run python scripts/benchmark_pronunciation.py --tts-engine mimo-tts,aliyun-tts --asr-engine aliyun-asr
```

The script writes `data/cache/benchmarks/pronunciation_benchmark.md` and `.csv`.
It creates clean, dropped-tail, dropped-middle, muted-middle, long-pause, slow,
global/local pitch-shifted, and noisy variants, then shows which score
dimensions react to each mistake.

## 📁 Repository layout

```
mandarin-pronunciation-coach/
├── README.md
├── pyproject.toml
├── uv.lock
├── requirements.txt                  # legacy reference
├── pytest.ini
├── config.py                         # all knobs in one place
├── data/
│   ├── reference_texts/              # your own reading material (txt)
│   ├── standard_audio/               # cached TTS reference WAVs
│   └── cache/                        # HF model cache
├── docs/
│   └── DESIGN.md                     # the original design write-up
├── scripts/
│   └── smoke_check.py                # live pipeline sanity check
├── src/
│   ├── audio/
│   │   ├── capture.py                # load / save audio
│   │   └── preprocess.py             # M1 — pre-emphasis, framing, VAD
│   ├── asr/
│   │   ├── models.py                 # lazy wav2vec2 holder
│   │   ├── recognizer.py             # M2 — CTC greedy ASR
│   │   └── aligner.py                # M2 — forced alignment + fallback
│   ├── features/
│   │   ├── spectral.py               # M3 — MFCC / FBank / energy / ZCR
│   │   └── pitch.py                  # M3 — F0 (pYIN) + helpers
│   ├── scoring/
│   │   ├── accuracy.py               # M4.1 — per-syllable + initial/final DTW on MFCC
│   │   ├── tone.py                   # M4.2 — Chao templates
│   │   ├── fluency.py                # M4.3 — VAD-based stats
│   │   ├── prosody.py                # M4.4 — contour DTW
│   │   ├── completeness.py           #        — CER
│   │   └── aggregator.py             # M4.5 — weighted overall
│   ├── feedback/
│   │   ├── tts.py                    # M5 — edge-tts / Aliyun / MiMo / F5-TTS
│   │   ├── visualize.py              # M5 — matplotlib figures
│   │   └── report.py                 # M5 — markdown / dict report
│   ├── app/
│   │   └── gradio_app.py             # web UI
│   └── pipeline.py                   # end-to-end orchestrator
└── tests/                            # 56 unit + integration tests
    ├── conftest.py
    ├── test_audio_preprocess.py
    ├── test_features.py
    ├── test_scoring.py
    ├── test_aligner_fallback.py
    └── test_pipeline_integration.py
```

## 🧮 Algorithm highlights

### Pre-emphasis + framing

`y[n] = x[n] - 0.97·x[n-1]`, then 25 ms frames with 10 ms hop and a Hamming
window.  Implemented from scratch in `src/audio/preprocess.py`.

### MFCC

Standard 13 static + Δ + ΔΔ = 39-dim features.  Used as the substrate for
the per-syllable accuracy DTW.

### Tone scoring (project differentiator)

The tone module now uses a reference-driven scoring path when a standard TTS
recording is available, and falls back to an interpretable feature classifier
when it is not.  The reference side can come from edge-tts, Aliyun Qwen-TTS,
or Xiaomi MiMo-V2.5-TTS.  We keep the iteration history because it is the
most useful story to explain in a course presentation.

**v1 — Chao five-scale template matching** *(turned out to fail on connected speech)*

Per-syllable: cut voiced F0 → z-score normalize → resample to 20 pts →
L2-distance to four Chao templates (55 / 35 / 214 / 51) → `argmin`.

Why it failed: per-syllable z-scoring kills absolute pitch position, so
**tone 1 (high level)** and **tone 3 (low level)** collapse to the same
"flat zero-mean" contour.  On real TTS / continuous speech every tone-1
syllable was misclassified, dragging the score below 25.

**v2 — Feature-based classifier with utterance baseline**

For each syllable we extract four interpretable F0 features:

| Feature | Computed as | What it captures |
|---|---|---|
| **Relative position** | `(median_st − utterance_median_st) / IQR` | High vs. low in the speaker's range — separates tone 1 from tone 3 |
| **Slope** | Linear `polyfit(t, F0_st)` coefficient | Rising (tone 2) vs falling (tone 4) |
| **Curvature** | Quadratic `polyfit` coefficient `a` | Positive `a` ⇒ U-shape ⇒ tone 3 dipping |
| **Range** | 90th − 10th percentile (semitones) | Distinguishes confident contours from noisy near-flat ones |

F0 is median-filtered first to remove pYIN octave glitches.  The decision
tree is ordered tone 3 → 4 → 2 → 1 → 5 to give the most distinctive shape
the first claim on each syllable.  Common confusable pairs
(1↔2, 2↔3, 1↔4, 3↔5) get 45 / 100 partial credit instead of a binary
correct / wrong flip — this prevents one borderline call from collapsing
the whole syllable's score.

On the smoke-check TTS sentence "今天天气真好。" v2 took the tone score from
**24.5 → 69.9** without changing any other module.

**v3 — Reference-driven F0 contour scoring** *(current when TTS reference exists)*

For each aligned character we also cut the corresponding F0 segment from the
standard TTS reference, such as MiMo-V2.5-TTS, Aliyun Qwen-TTS, or edge-tts.
Both curves are converted to semitones and centered by their own utterance-level pitch
baseline.  Then we compute three interpretable sub-scores:

| Sub-score | Method | Purpose |
|---|---|---|
| **Contour score** | DTW distance between normalized F0 contours | Captures the actual tone melody |
| **Slope score** | Difference between start-to-end F0 movement | Separates rising / falling behaviour |
| **Coverage score** | Whether both sides have enough voiced F0 frames | Penalizes unreliable windows |

The final tone score is a weighted fusion of these three scores.  The
standalone classifier is still shown as a diagnostic label, but the numeric
score is mainly driven by user-vs-reference F0 similarity.

When multiple reference voices are configured, for example `mimo-tts,aliyun-tts`,
the pipeline evaluates each reference independently and keeps the better
per-character match for accuracy and tone.  This reduces false penalties caused
by one TTS voice's speaking style or sentence intonation.

With Aliyun Qwen-TTS as both the "user" audio and reference audio for
"今天天气真好。", v3 raises the tone dimension from **58.97 → 98.96**.  This is
the expected behaviour: identical high-quality reference speech should not be
penalized just because a rule-based classifier calls a connected-speech tone
borderline.

### DTW accuracy

We synthesize a TTS reference for the *same text*, extract MFCC, slice both
the user and reference recordings by the per-syllable alignment, and compute
cosine-distance DTW (via `librosa.sequence.dtw`).  Mean-per-step cost is
squashed into a 0..100 score.

The current version also estimates a pinyin-based initial/final boundary for
each character.  It runs the same MFCC-DTW comparison on the initial segment
and final segment separately, so the report can say whether a low character
accuracy score is closer to a 声母偏差 or 韵母偏差.

Accuracy also fuses an articulation cue: the user's voiced-frame coverage and
local duration are compared with the reference window.  This catches local
silence, swallowed syllables, and very short alignment windows that MFCC-DTW
alone can sometimes score too generously.

### Completeness localization

Completeness still uses character-level CER for the whole utterance.  In
addition, the scorer aligns the recognized text and reference text with a
longest-common-subsequence pass, then marks each reference character as 已读
or 漏读.  The per-character report can therefore show exactly which character
is suspected to be missing.

### Confidence estimation

The system estimates score reliability separately from the score itself.  It
combines five evidence signals: valid speech duration, TTS reference
availability, ASR coverage, F0 availability, and per-character articulation
coverage.  The final report shows an overall confidence score plus per-character
高 / 中 / 低 confidence labels.  This is useful when the recording is too short,
too noisy, missing ASR output, or lacks a standard reference voice.

### Fluency scoring

Fluency now uses a PSC-style reading model instead of a narrow speed target.
The scoring gives full credit to a natural reading-rate band, allows a small
amount of short-sentence demonstration pause, and adds a rhythm-stability score
from per-character duration variance.  This prevents formal standard readings
from being unfairly penalized while still catching long silences and choppy
reading.

### Error-simulation benchmark

`scripts/benchmark_pronunciation.py` creates controlled variants of a standard
reading.  In the default run, deleting the tail or middle lowers completeness,
muting a local segment hurts local accuracy, inserting a long pause lowers
fluency, global/local pitch-shifting lowers tone, and noise lowers tone /
prosody.  This gives a compact experiment table for explaining that the five
score dimensions respond to different pronunciation problems.

### Forced alignment

Primary path: `torchaudio.functional.forced_align` driven by log-probs from
`jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`.  Frame rate ≈ 50 Hz
(20 ms per frame).  We collapse consecutive same-token runs in the alignment
path to recover (start_frame, end_frame) for each reference character.

Fallback path: if the model fails to load or too many characters are OOV, we
distribute characters uniformly across all VAD-voiced regions.  Coarse, but
keeps tone/fluency scoring usable.

## 🧪 Testing strategy

| Layer | Test file | What it covers |
|---|---|---|
| DSP | `test_audio_preprocess.py` | Pre-emphasis high-freq boost, frame shape, VAD voiced vs silent |
| Features | `test_features.py` | MFCC shape, energy ordering, pYIN recovers known 220 Hz pitch |
| Scoring | `test_scoring.py` | Tone classification on synthetic flat / rising / dipping / falling contours, accuracy DTW, initial/final sub-syllable scores, fluency stats, aggregator |
| Alignment | `test_aligner_fallback.py` | VAD-fallback aligner produces monotone timestamps |
| Pipeline | `test_pipeline_integration.py` | End-to-end smoke (no model load) |

The full suite runs in ~3.5 min on CPU.  The wav2vec2 model is **only**
loaded by the live `scripts/smoke_check.py` and the Gradio app, never by the
tests, so CI stays fast.

## ⚖️ Known limitations

| Limitation | Why | Workaround |
|---|---|---|
| Neutral tone (轻声) detection is heuristic | F0-only signal is too short / context-dependent | We give benefit-of-the-doubt when the contour is low + short |
| Complex tone sandhi is still heuristic | Common rules like 三声连读 and 一/不 变调 are handled, but phrase-level sandhi is broader | The report shows both dictionary tone and scoring tone |
| Erhua (儿化) treated as separate char | The current aligner works at character level | Mark as future work |
| `jonatasgrosman` model has ~3.5k char vocab | Some rare characters are OOV | Aligner skips OOV chars and interpolates timestamps |
| TTS reference is not a "PSC examiner" voice | edge-tts, Aliyun Qwen-TTS, and MiMo-V2.5-TTS are neural voices | Use multiple reference voices for robustness checks |

## 📚 References

- 普通话水平测试 (PSC) 测试大纲
- Chao Y.R. — *A System of Tone Letters*, 1930
- Hsu, W. et al. — *Wav2Vec 2.0*
- Witt S. — *Use of speech recognition in CALL*, 1999 (GOP)
- SpeechOcean762 dataset (OpenSLR #101) — *used as methodological validation*; note that it is **English L2** data, not Mandarin

## 📄 License

MIT.  See `LICENSE`.
