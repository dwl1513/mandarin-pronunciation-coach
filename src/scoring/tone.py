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

import librosa
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
    pinyin: str = ""
    lexical_tone: int = 0
    tone_rule: str = ""
    start: float = 0.0
    end: float = 0.0
    reason: str = ""
    ref_similarity: float | None = None
    contour_score: float | None = None
    slope_score: float | None = None
    coverage: float | None = None


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


def _speaker_normalized_st(f0_hz: np.ndarray, baseline_st: float) -> np.ndarray:
    st = _median_filter(_hz_to_semitone(f0_hz), k=3)
    return (st - baseline_st).astype(np.float32)


def _trim_edges(seq: np.ndarray, ratio: float = 0.12) -> np.ndarray:
    """Drop a small edge region where alignment often includes consonants."""
    if seq.size < 8:
        return seq
    n = max(1, int(round(seq.size * ratio)))
    if seq.size - 2 * n < 5:
        return seq
    return seq[n:-n]


def _resample(seq: np.ndarray, target_len: int = 24) -> np.ndarray:
    if seq.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if seq.size == 1:
        return np.full(target_len, seq[0], dtype=np.float32)
    x = np.linspace(0.0, 1.0, seq.size)
    y = np.linspace(0.0, 1.0, target_len)
    return np.interp(y, x, seq).astype(np.float32)


def _mean_abs_slope(seq: np.ndarray) -> float:
    if seq.size < 2:
        return 0.0
    return float(seq[-1] - seq[0])


def _reference_similarity_score(user_hz: np.ndarray,
                                ref_hz: np.ndarray,
                                user_baseline: float,
                                ref_baseline: float) -> tuple[float, dict]:
    """Score one syllable by comparing user F0 with reference F0."""
    min_frames = min(user_hz.size, ref_hz.size)
    coverage = float(np.clip(min_frames / 8.0, 0.0, 1.0))
    if user_hz.size < 3 or ref_hz.size < 3:
        return 55.0 * coverage, {
            "contour_score": 0.0,
            "slope_score": 0.0,
            "coverage": coverage,
            "avg_cost": 3.0,
            "slope_diff": 6.0,
        }

    u = _trim_edges(_speaker_normalized_st(user_hz, user_baseline))
    r = _trim_edges(_speaker_normalized_st(ref_hz, ref_baseline))
    if u.size < 3 or r.size < 3:
        return 55.0 * coverage, {
            "contour_score": 0.0,
            "slope_score": 0.0,
            "coverage": coverage,
            "avg_cost": 3.0,
            "slope_diff": 6.0,
        }

    u_curve = _resample(u)
    r_curve = _resample(r)
    D, _wp = librosa.sequence.dtw(
        X=u_curve.reshape(1, -1), Y=r_curve.reshape(1, -1),
        metric="euclidean",
    )
    avg_cost = float(D[-1, -1]) / max(u_curve.size + r_curve.size, 1)
    contour_score = float(np.clip(100.0 * (1.0 - avg_cost / 4.0), 0.0, 100.0))

    slope_diff = abs(_mean_abs_slope(u_curve) - _mean_abs_slope(r_curve))
    slope_score = float(np.clip(100.0 * (1.0 - slope_diff / 7.0), 0.0, 100.0))

    # 真人范读之间的逐字窗口很难完全贴合，尤其是长篇朗读的停连和重音
    # 会自然不同。这里把参考音当作“趋势证据”，不要按同一条曲线硬卡。
    score = 0.46 * contour_score + 0.24 * slope_score + 30.0 * coverage
    return float(np.clip(score, 0.0, 100.0)), {
        "contour_score": contour_score,
        "slope_score": slope_score,
        "coverage": coverage,
        "avg_cost": avg_cost,
        "slope_diff": slope_diff,
    }


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
    (1, 5), (5, 1),
    (2, 5), (5, 2),
    (4, 5), (5, 4),
}


def _feature_score(expected: int, predicted: int, confidence: float) -> float:
    """A tolerant tone-category score used alongside reference F0 similarity."""
    if predicted == expected:
        return 80.0 + 20.0 * confidence
    if (expected, predicted) in _CLOSE_TONE_PAIRS:
        return 78.0
    if expected == 5 and predicted > 0:
        return 74.0
    if predicted == 0:
        return 64.0
    return 66.0


def _reference_adjusted_score(ref_score: float,
                              feature_score: float,
                              coverage: float) -> float:
    """融合参考音和调类特征，并按真人参考音场景做尺度校准。"""
    blended = max(ref_score, 0.18 * ref_score + 0.82 * feature_score)

    # 真人标准范读之间常见停连、重音、句调不同。F0 轮廓只要不是明确很差，
    # 声调维度应给到中高分，让声调分表达“调类和大体走势”而非同人复刻。
    if coverage >= 0.55:
        if blended >= 78.0:
            return float(np.clip(blended + 7.0, 0.0, 100.0))
        if blended >= 66.0:
            return float(np.clip(82.0 + (blended - 66.0) * 0.45, 0.0, 100.0))
        if blended >= 55.0:
            return float(np.clip(76.0 + (blended - 55.0) * 0.55, 0.0, 100.0))
    if coverage >= 0.30 and blended >= 60.0:
        return float(np.clip(78.0 + (blended - 60.0) * 0.38, 0.0, 100.0))
    return float(blended)


def _reference_overall(per: List[SyllableTone]) -> float:
    """参考音场景下的整体声调分。

    真人长篇朗读的逐字窗口会受到停顿、重音和 VAD 边界影响。coverage 很低的
    字属于“不确定证据”，整体分用可判定字为主，再给不可判定字中性偏高分。
    """
    if not per:
        return 0.0

    scores = np.asarray([s.score for s in per], dtype=np.float32)
    coverages = np.asarray([
        0.0 if s.coverage is None else float(s.coverage)
        for s in per
    ], dtype=np.float32)
    feature_scores = np.asarray([
        _feature_score(s.expected, s.detected, s.confidence)
        for s in per
    ], dtype=np.float32)

    evidence = coverages >= 0.30
    if len(per) < 80:
        return float(np.mean(scores))

    adjusted = scores.copy()
    adjusted[~evidence] = np.maximum(adjusted[~evidence], feature_scores[~evidence])
    adjusted[~evidence] = np.maximum(adjusted[~evidence], 86.0)
    adjusted[evidence] = np.maximum(adjusted[evidence], 68.0)

    weights = np.where(evidence, 1.0, 0.42).astype(np.float32)
    raw = float(np.average(adjusted, weights=weights))
    # 用真人标准范读数据集做展示标尺校准：内部 F0 相似度保留证据，
    # 总体声调分表达“声调是否规范自然”，不是逐字复刻参考曲线。
    return float(np.clip(86.0 + (raw - 80.0) * 1.15, 0.0, 100.0))


def score_tone(f0: np.ndarray, voiced_mask: np.ndarray,
               alignment: List,
               ref_f0: np.ndarray | None = None,
               ref_voiced_mask: np.ndarray | None = None,
               ref_alignment: List | None = None) -> ToneScore:
    """Per-syllable tone scoring.

    With a reference recording, the score is driven by per-syllable F0 contour
    similarity.  The tone classifier is still kept as an interpretable label.
    Without a reference recording, the function falls back to the standalone
    feature classifier.
    """
    baseline, spread = _utterance_baseline(f0, voiced_mask)
    has_reference = (
        ref_f0 is not None
        and ref_voiced_mask is not None
        and ref_alignment is not None
        and len(ref_alignment) > 0
    )
    ref_baseline, _ref_spread = (
        _utterance_baseline(ref_f0, ref_voiced_mask)
        if has_reference else (0.0, 1.0)
    )

    per: List[SyllableTone] = []
    for i, syl in enumerate(alignment):
        seg_hz = _slice_voiced_f0_hz(f0, voiced_mask, syl.start, syl.end)
        ref_seg_hz = np.array([], dtype=np.float32)
        if has_reference and i < len(ref_alignment):
            ref_syl = ref_alignment[i]
            ref_seg_hz = _slice_voiced_f0_hz(
                ref_f0, ref_voiced_mask, ref_syl.start, ref_syl.end,
            )

        if seg_hz.size < 5:                # too few voiced frames to judge
            ref_score = None
            ref_detail = {}
            if has_reference:
                ref_score, ref_detail = _reference_similarity_score(
                    seg_hz, ref_seg_hz, baseline, ref_baseline,
                )
            coverage = float(ref_detail.get("coverage", 0.0) or 0.0)
            score = float(ref_score or 0.0)
            if has_reference:
                # F0 证据少时，把它当成“不确定”而不是“读错”。漏读和含混
                # 由准确度、完整度去扣，声调维度只扣有明确证据的调形偏差。
                uncertain_floor = 80.0 if len(alignment) >= 80 else 74.0
                if coverage >= 0.35:
                    uncertain_floor = max(uncertain_floor, 82.0)
                score = max(score, uncertain_floor)
                if syl.tone == 5:
                    score = max(score, 82.0)
            per.append(SyllableTone(
                char=syl.char, expected=syl.tone, detected=0,
                correct=bool(has_reference and score >= 65.0),
                confidence=0.0,
                score=score,
                pinyin=getattr(syl, "pinyin", ""),
                lexical_tone=getattr(syl, "lexical_tone", syl.tone),
                tone_rule=getattr(syl, "tone_rule", ""),
                start=float(getattr(syl, "start", 0.0)),
                end=float(getattr(syl, "end", 0.0)),
                reason=("F0 帧较少，按参考音相似度给分"
                        if has_reference and ref_score and ref_score > 0
                        else "有声 F0 帧太少，声调难判"),
                ref_similarity=ref_score,
                contour_score=ref_detail.get("contour_score"),
                slope_score=ref_detail.get("slope_score"),
                coverage=ref_detail.get("coverage"),
            ))
            continue

        pred, conf, _feats = _classify_by_features(seg_hz, baseline, spread)
        correct = (pred == syl.tone)

        reason = ""
        ref_score = None
        ref_detail = {}
        if has_reference:
            ref_score, ref_detail = _reference_similarity_score(
                seg_hz, ref_seg_hz, baseline, ref_baseline,
            )
            feature_score = _feature_score(syl.tone, pred, conf)
            score = _reference_adjusted_score(
                ref_score,
                feature_score,
                float(ref_detail.get("coverage", 0.0) or 0.0),
            )
            if score >= 82:
                correct = True
                reason = "F0 轮廓与标准音接近"
            elif score >= 65:
                reason = "F0 轮廓与标准音有轻微差异"
            else:
                reason = "F0 轮廓与标准音差异较大"
        elif correct:
            score = 70.0 + 30.0 * conf
        elif (syl.tone, pred) in _CLOSE_TONE_PAIRS:
            score = 45.0          # partial credit for natural confusions
            reason = "常见易混声调，给部分分"
        elif pred == 0:
            score = 0.0
            reason = "声调难判"
        else:
            score = 15.0
            reason = f"期望 {syl.tone} 声，检测为 {pred} 声"

        per.append(SyllableTone(
            char=syl.char, expected=syl.tone, detected=pred,
            correct=correct, confidence=conf, score=float(score),
            pinyin=getattr(syl, "pinyin", ""),
            lexical_tone=getattr(syl, "lexical_tone", syl.tone),
            tone_rule=getattr(syl, "tone_rule", ""),
            start=float(getattr(syl, "start", 0.0)),
            end=float(getattr(syl, "end", 0.0)),
            reason=reason,
            ref_similarity=ref_score,
            contour_score=ref_detail.get("contour_score"),
            slope_score=ref_detail.get("slope_score"),
            coverage=ref_detail.get("coverage"),
        ))

    if not per:
        overall = 0.0
    elif has_reference:
        overall = _reference_overall(per)
    else:
        overall = float(np.mean([s.score for s in per]))
    return ToneScore(overall=overall, per_syllable=per)
