"""Tone scoring (M4.2) — the project's signature contribution.

We classify each syllable as one of the four lexical tones (+ neutral) using
**three intuitive features** of its F0 contour, computed on top of the
utterance-level pitch baseline:

    1. Median position    — F0 above / below the speaker's mid-pitch
                            (separates tone 1 high-level from tone 3 low).
    2. Linear slope       — overall rise (tone 2) vs fall (tone 4) in
                            semitones across the syllable.
    3. Curvature          — positive (U-shape) detects the tone 3 dipping
                            contour even when the dip isn't very deep.

Why not just match the textbook Chao 55/35/214/51 templates?  Because
templates assume *isolated syllables*.  In connected speech, declination and
sentence intonation deform individual tones, and per-syllable z-scoring
throws away the absolute pitch position — making tone 1 indistinguishable
from tone 3.  The feature-based classifier keeps the absolute position
(relative to the utterance baseline) and is therefore much more robust on
real speech / TTS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from config import HOP_LENGTH, SAMPLE_RATE


@dataclass
class SyllableTone:
    char: str
    expected: int           # ground-truth tone from pinyin
    detected: int           # 0 if undetectable (unvoiced / too short)
    correct: bool
    confidence: float       # 0..1 — how strong the winning feature signal was
    score: float            # 0..100


@dataclass
class ToneScore:
    overall: float
    per_syllable: List[SyllableTone] = field(default_factory=list)


# ------------------------------------------------------------------- helpers
def _hz_to_semitone(hz: np.ndarray, ref: float = 100.0) -> np.ndarray:
    return 12.0 * np.log2(np.maximum(hz, 1e-3) / ref)


def _median_filter(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Cheap median filter (k must be odd) — removes pYIN octave glitches."""
    if x.size <= k:
        return x
    pad = k // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = np.median(padded[i:i + k])
    return out


def _slice_voiced_f0_hz(f0: np.ndarray, voiced_mask: np.ndarray,
                         start_s: float, end_s: float,
                         hop: int = HOP_LENGTH, sr: int = SAMPLE_RATE
                         ) -> np.ndarray:
    """Voiced F0 (Hz) inside the alignment window."""
    frame_dur = hop / sr
    s = max(0, int(np.floor(start_s / frame_dur)))
    e = min(len(f0), int(np.ceil(end_s / frame_dur)) + 1)
    seg = f0[s:e]
    mask = voiced_mask[s:e] & (seg > 0)
    return seg[mask]


def _utterance_baseline(f0: np.ndarray, voiced_mask: np.ndarray
                         ) -> Tuple[float, float]:
    """Speaker's pitch centre + spread across the whole utterance, semitones."""
    voiced = f0[voiced_mask & (f0 > 0)]
    if voiced.size < 5:
        return 0.0, 1.0
    st = _hz_to_semitone(voiced)
    # Trimmed median is robust to a handful of doubling/halving errors.
    median = float(np.median(st))
    # Use 75–25 inter-quartile range as a robust "speaker spread".
    iqr = float(np.percentile(st, 75) - np.percentile(st, 25))
    return median, max(iqr, 1.0)


# ----------------------------------------------------- the feature classifier
def _classify_by_features(syl_f0_hz: np.ndarray,
                           baseline_st: float, spread_st: float
                           ) -> Tuple[int, float, dict]:
    """Predict tone from {position, slope, curvature, range}. Returns
    (predicted_tone, confidence_0_1, raw_features_for_debug)."""
    n = syl_f0_hz.size
    if n < 5:
        return 0, 0.0, {}

    # Median-filter the F0 trace inside the syllable to suppress octave
    # errors before any polynomial fitting.
    st = _median_filter(_hz_to_semitone(syl_f0_hz), k=3)

    # Position relative to speaker baseline — positive = high in range.
    rel_pos = (float(np.median(st)) - baseline_st) / spread_st

    # Linear and quadratic fits over normalized time x ∈ [0, 1].
    x = np.linspace(0.0, 1.0, n)
    slope = float(np.polyfit(x, st, 1)[0])          # semitones per syllable
    a, _b, _c = np.polyfit(x, st, 2)                 # ax² + bx + c
    curvature = float(a)                             # > 0  ⇒ U-shape (dip)

    # Pitch range inside the syllable (robust 10-90 percentile gap).
    rng = float(np.percentile(st, 90) - np.percentile(st, 10))

    feats = {"rel_pos": rel_pos, "slope": slope,
             "curvature": curvature, "range": rng}

    # ---- decision logic, ordered by how distinctive each tone is. ----------
    # Tone 3 — dipping. The curvature must be positive AND the contour must
    # actually span enough range to count as a dip (not just noisy flatness).
    if curvature > 1.5 and rng > 1.5 and slope < 2.0:
        conf = float(np.clip(curvature / 4.0, 0.3, 1.0))
        return 3, conf, feats

    # Tone 4 — clear fall. Strong negative slope, decent range.
    if slope < -1.5 and rng > 1.5:
        conf = float(np.clip(-slope / 4.0, 0.3, 1.0))
        return 4, conf, feats

    # Tone 2 — clear rise. Strong positive slope.
    if slope > 1.5 and rng > 1.5:
        conf = float(np.clip(slope / 4.0, 0.3, 1.0))
        return 2, conf, feats

    # Tone 1 — flat AND high. Slope close to zero, position not low.
    if abs(slope) <= 2.0 and rng < 3.0 and rel_pos > -0.4:
        flatness = 1.0 - min(1.0, abs(slope) / 2.0)
        height   = float(np.clip((rel_pos + 0.4) / 1.0, 0.2, 1.0))
        return 1, float(0.5 * flatness + 0.5 * height), feats

    # Tone 5 (neutral) — short / flat / low. Catch-all for soft endings.
    if rng < 2.0 and rel_pos < 0.0:
        return 5, 0.5, feats

    # Fallback — let slope sign cast a weak vote.
    if slope > 0:
        return 2, 0.25, feats
    if slope < 0:
        return 4, 0.25, feats
    return 1, 0.25, feats


# ------------------------------------------------------------- public API
_CLOSE_TONE_PAIRS = {           # commonly-confused pairs → partial credit
    (1, 2), (2, 1),
    (2, 3), (3, 2),
    (1, 4), (4, 1),
    (3, 5), (5, 3),             # neutral often looks like a soft tone 3
}


def score_tone(f0: np.ndarray, voiced_mask: np.ndarray,
               alignment: List) -> ToneScore:
    """Per-syllable tone classification + 0..100 scoring."""
    baseline, spread = _utterance_baseline(f0, voiced_mask)

    per: List[SyllableTone] = []
    for syl in alignment:
        seg_hz = _slice_voiced_f0_hz(f0, voiced_mask, syl.start, syl.end)
        if seg_hz.size < 5:                # too few voiced frames to judge
            per.append(SyllableTone(
                char=syl.char, expected=syl.tone, detected=0,
                correct=False, confidence=0.0, score=0.0,
            ))
            continue

        pred, conf, _feats = _classify_by_features(seg_hz, baseline, spread)
        correct = (pred == syl.tone)

        if correct:
            score = 70.0 + 30.0 * conf
        elif (syl.tone, pred) in _CLOSE_TONE_PAIRS:
            score = 45.0          # partial credit for natural confusions
        elif pred == 0:
            score = 0.0
        else:
            score = 15.0

        per.append(SyllableTone(
            char=syl.char, expected=syl.tone, detected=pred,
            correct=correct, confidence=conf, score=float(score),
        ))

    overall = float(np.mean([s.score for s in per])) if per else 0.0
    return ToneScore(overall=overall, per_syllable=per)
