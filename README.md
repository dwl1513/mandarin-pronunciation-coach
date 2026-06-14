# 🗣️ Mandarin Pronunciation Coach · 普通话 AI 发音教练

[![python](https://img.shields.io/badge/python-3.10-blue.svg)]()
[![tests](https://img.shields.io/badge/tests-36%20passing-brightgreen.svg)]()
[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()
an
end-to-end Mandarin pronunciation assessment + correction system.  The user
reads (or speaks freely) a given text into the microphone; the system scores
their pronunciation along five dimensions and gives **per-character** feedback
plus a TTS-generated standard reference.

## ✨ Features

- **5-dim scoring** aligned with the official 普通话水平测试 (PSC) rubric
  - 声韵母准确度 (DTW on MFCC vs. TTS reference)
  - **声调** — Chao five-scale templates + normalized F0 contour matching (project differentiator)
  - 流利度 (语速 / 停顿 / 停顿占比)
  - 韵律自然度 (sentence-level F0 contour DTW + pitch range)
  - 完整度 (CER against ASR transcription)
- **Per-character diagnosis** with tone-error highlighting
- **Standard pronunciation playback** via Microsoft edge-tts (free, online) with optional F5-TTS for voice-cloned references
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
| `src/asr/recognizer.py` | ASR for completeness | Pretrained wav2vec2-CTC |
| `src/asr/aligner.py` | Forced alignment + VAD fallback | torchaudio.functional.forced_align |
| `src/scoring/accuracy.py` | DTW MFCC matching | **From scratch (algorithm)** |
| `src/scoring/tone.py` | Chao template tone classifier | **From scratch (algorithm)** |
| `src/scoring/fluency.py` | Speech rate / pause statistics | **From scratch (algorithm)** |
| `src/scoring/prosody.py` | Contour DTW + pitch range | **From scratch (algorithm)** |
| `src/scoring/completeness.py` | Character-level CER | jiwer |
| `src/feedback/tts.py` | Reference synthesis | edge-tts / F5-TTS |
| `src/app/gradio_app.py` | Web UI | Gradio |

## 🚀 Quick start

### 1. Environment

The project is developed in a conda env called **`f5tts`** that already has
`torch + torchaudio + transformers + librosa + pypinyin + scikit-learn +
pytest` installed (because F5-TTS reuses the same stack).  Activate it and
install the few extras:

```powershell
conda activate f5tts
pip install -r requirements.txt
```

If you don't already have the `f5tts` env, create a fresh one:

```powershell
conda create -n mpc python=3.10 -y
conda activate mpc
pip install -r requirements.txt
```

> **Windows note** — Microsoft YaHei is used for plot labels.  If you're on
> Linux, install a CJK font (e.g. `fonts-noto-cjk`) and matplotlib will pick
> it up automatically.

### 2. Run the unit tests (≈3.5 min, no network / no model download)

```powershell
python -m pytest -q
```

> ⚠️ **Use `python -m pytest`, not bare `pytest`.** On Windows with multiple
> conda envs that have `pytest` installed, a bare `pytest` may resolve to a
> different env's executable (PATH ordering), causing `ModuleNotFoundError`
> for librosa / pypinyin even after `conda activate f5tts`. `python -m
> pytest` always uses the currently-active Python.

You should see `36 passed`.

### 3. Launch the Gradio app

```powershell
python -m src.app.gradio_app
```

The first time you press "开始评测" the system will:

1. download the wav2vec2 Chinese ASR/alignment checkpoint (~1.2 GB) — cached
   under `data/cache/hf/`;
2. ping edge-tts to synthesize the reference audio — cached under
   `data/standard_audio/`.

Subsequent runs are fast and fully offline (except the very first edge-tts
call for any new sentence).

### 4. CLI smoke check (optional)

```powershell
python scripts/smoke_check.py
```

This runs TTS → preprocess → align → all 5 scorers against a built-in
sentence and prints the result.  Use it to verify the live model + network
path on a new machine.

## 📁 Repository layout

```
mandarin-pronunciation-coach/
├── README.md
├── requirements.txt
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
│   │   ├── accuracy.py               # M4.1 — DTW on MFCC
│   │   ├── tone.py                   # M4.2 — Chao templates
│   │   ├── fluency.py                # M4.3 — VAD-based stats
│   │   ├── prosody.py                # M4.4 — contour DTW
│   │   ├── completeness.py           #        — CER
│   │   └── aggregator.py             # M4.5 — weighted overall
│   ├── feedback/
│   │   ├── tts.py                    # M5 — edge-tts / F5-TTS
│   │   ├── visualize.py              # M5 — matplotlib figures
│   │   └── report.py                 # M5 — markdown / dict report
│   ├── app/
│   │   └── gradio_app.py             # web UI
│   └── pipeline.py                   # end-to-end orchestrator
└── tests/                            # 36 unit + integration tests
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

### Tone classification (project differentiator)

The classifier went through two iterations — the v2 design is what the code
actually runs today.  We kept the iteration history because the v1 → v2
story is itself the most pedagogically interesting part of the project.

**v1 — Chao five-scale template matching** *(turned out to fail on connected speech)*

Per-syllable: cut voiced F0 → z-score normalize → resample to 20 pts →
L2-distance to four Chao templates (55 / 35 / 214 / 51) → `argmin`.

Why it failed: per-syllable z-scoring kills absolute pitch position, so
**tone 1 (high level)** and **tone 3 (low level)** collapse to the same
"flat zero-mean" contour.  On real TTS / continuous speech every tone-1
syllable was misclassified, dragging the score below 25.

**v2 — Feature-based classifier with utterance baseline** *(current)*

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

### DTW accuracy

We synthesize a TTS reference for the *same text*, extract MFCC, slice both
the user and reference recordings by the per-syllable alignment, and compute
cosine-distance DTW (via `librosa.sequence.dtw`).  Mean-per-step cost is
squashed into a 0..100 score.

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
| Scoring | `test_scoring.py` | Tone classification on synthetic flat / rising / dipping / falling contours, accuracy DTW on identical inputs, fluency stats, aggregator |
| Alignment | `test_aligner_fallback.py` | VAD-fallback aligner produces monotone timestamps |
| Pipeline | `test_pipeline_integration.py` | End-to-end smoke (no model load) |

The full suite runs in ~3.5 min on CPU.  The wav2vec2 model is **only**
loaded by the live `scripts/smoke_check.py` and the Gradio app, never by the
tests, so CI stays fast.

## ⚖️ Known limitations

| Limitation | Why | Workaround |
|---|---|---|
| Neutral tone (轻声) detection is heuristic | F0-only signal is too short / context-dependent | We give benefit-of-the-doubt when the contour is low + short |
| Tone sandhi (3+3→2+3, 一/不 变调) not handled | Reference tones come from pypinyin lookup, not actual sentence context | Mark as future work; PSC scoring is lenient on this |
| Erhua (儿化) treated as separate char | Same | Same |
| `jonatasgrosman` model has ~3.5k char vocab | Some rare characters are OOV | Aligner skips OOV chars and interpolates timestamps |
| TTS reference is not a "PSC examiner" voice | edge-tts uses a Microsoft neural voice | Acceptable for DTW / prosody comparison; can be swapped via `config.EDGE_TTS_VOICE` |

## 📚 References

- 普通话水平测试 (PSC) 测试大纲
- Chao Y.R. — *A System of Tone Letters*, 1930
- Hsu, W. et al. — *Wav2Vec 2.0*
- Witt S. — *Use of speech recognition in CALL*, 1999 (GOP)
- SpeechOcean762 dataset (OpenSLR #101) — *used as methodological validation*; note that it is **English L2** data, not Mandarin

## 📄 License

MIT.  See `LICENSE` (not included by default — add one before publishing).
