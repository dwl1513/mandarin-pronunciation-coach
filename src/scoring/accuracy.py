"""基于 DTW 的发音准确度评分 (M4.1)。

每个对齐音节都会从用户音频和标准音频中取出对应 MFCC 窗口，再用动态时间
规整计算两段特征的距离。距离越小，说明越接近标准音，得分越高。

除整字分外，这里还估计声母段和韵母段的子音节分数。这样仍能保持轻量，
同时让逐字报告看到偏差更像出在字头还是字腹/字尾。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import librosa
import numpy as np

from config import HOP_LENGTH, SAMPLE_RATE

_INITIALS = (
    "zh", "ch", "sh",
    "b", "p", "m", "f",
    "d", "t", "n", "l",
    "g", "k", "h",
    "j", "q", "x",
    "r", "z", "c", "s",
)


@dataclass
class SyllableAccuracy:
    char: str
    score: float            # 0..100
    dtw_cost: float         # raw mean-cost from DTW
    duration: float         # syllable duration in seconds
    initial_score: Optional[float] = None
    final_score: Optional[float] = None
    initial_cost: Optional[float] = None
    final_cost: Optional[float] = None
    initial: str = ""
    final: str = ""
    articulation_score: Optional[float] = None
    voiced_coverage_score: Optional[float] = None
    duration_score: Optional[float] = None
    user_voiced_ratio: Optional[float] = None
    ref_voiced_ratio: Optional[float] = None


@dataclass
class AccuracyScore:
    overall: float                                           # 0..100
    per_syllable: List[SyllableAccuracy] = field(default_factory=list)


# --------------------------------------------------------------- helpers
def _slice_mfcc(mfcc: np.ndarray, start_s: float, end_s: float,
                hop: int = HOP_LENGTH, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Cut MFCC frames whose centre time falls in [start_s, end_s]."""
    if mfcc.size == 0:
        return mfcc
    frame_dur = hop / sr
    s_idx = max(0, int(np.floor(start_s / frame_dur)))
    e_idx = min(len(mfcc), int(np.ceil(end_s / frame_dur)) + 1)
    if e_idx - s_idx < 2:
        # pad to at least 2 frames so DTW behaves
        s_idx = max(0, e_idx - 2)
    return mfcc[s_idx:e_idx]


def _slice_mask(mask: np.ndarray,
                start_s: float,
                end_s: float,
                hop: int = HOP_LENGTH,
                sr: int = SAMPLE_RATE) -> np.ndarray:
    """按时间窗取帧级布尔掩码。"""
    if mask.size == 0 or end_s <= start_s:
        return np.asarray([], dtype=bool)
    frame_dur = hop / sr
    s_idx = max(0, int(np.floor(start_s / frame_dur)))
    e_idx = min(len(mask), int(np.ceil(end_s / frame_dur)) + 1)
    return mask[s_idx:e_idx].astype(bool, copy=False)


def _dtw_mean_cost(a: np.ndarray, b: np.ndarray) -> float:
    """librosa DTW mean-per-step cost between two [T, D] sequences."""
    if a.size == 0 or b.size == 0:
        return 1.0
    # librosa expects [D, T]
    D, _wp = librosa.sequence.dtw(X=a.T, Y=b.T, metric="cosine")
    # Mean cost along the warp path: trace back from corner
    path_cost = float(D[-1, -1])
    path_len = (a.shape[0] + b.shape[0])
    return path_cost / max(path_len, 1)


def _cost_to_score(cost: float) -> float:
    """Map [0, ~1] cosine DTW cost to a 0..100 score (monotone decreasing)."""
    # Empirically: cost ≈ 0.1 → great match, ≈ 0.5 → poor.
    # Use a smooth squashing so small differences near 0 don't dominate.
    cost = float(np.clip(cost, 0.0, 1.5))
    score = 100.0 * (1.0 - cost / 0.6)
    return float(np.clip(score, 0.0, 100.0))


def _voiced_ratio(mask: np.ndarray | None,
                  start_s: float,
                  end_s: float) -> Optional[float]:
    """计算音节窗口内的有效发声比例。"""
    if mask is None:
        return None
    seg = _slice_mask(mask, start_s, end_s)
    if seg.size == 0:
        return None
    return float(np.mean(seg))


def _voiced_coverage_score(user_ratio: Optional[float],
                           ref_ratio: Optional[float]) -> Optional[float]:
    """比较用户和标准音在同一字上的有效发声覆盖。"""
    if user_ratio is None or ref_ratio is None:
        return None
    if ref_ratio < 0.12:
        return None
    expected = max(0.12, ref_ratio * 0.8)
    return float(np.clip(100.0 * user_ratio / expected, 0.0, 100.0))


def _duration_score(user_duration: float, ref_duration: float) -> Optional[float]:
    """惩罚明显短到像漏读的局部对齐窗口。"""
    if ref_duration <= 0.03:
        return None
    ratio = user_duration / max(ref_duration, 1e-6)
    if ratio >= 0.55:
        return 100.0
    return float(np.clip(100.0 * ratio / 0.55, 0.0, 100.0))


def _combine_accuracy(dtw_score: float,
                      voiced_score: Optional[float],
                      dur_score: Optional[float]) -> tuple[float, Optional[float]]:
    """融合谱形相似、有效发声覆盖和局部时长完整性。"""
    cues = [v for v in (voiced_score, dur_score) if v is not None]
    if not cues:
        return dtw_score, None
    articulation = float(min(cues))
    score = 0.72 * dtw_score + 0.28 * articulation
    if articulation < 60.0:
        score = min(score, articulation + 25.0)
    return float(np.clip(score, 0.0, 100.0)), articulation


def _split_pinyin(pinyin: str) -> tuple[str, str]:
    """从无声调拼音里估计声母和韵母标签。"""
    py = (pinyin or "").lower().replace("ü", "v")
    for initial in _INITIALS:
        if py.startswith(initial):
            return initial, py[len(initial):] or "∅"
    return "零声母", py or "∅"


def _initial_ratio(initial: str, syllable_duration: float) -> float:
    """Estimate the initial/final boundary inside a syllable window.

    普通话声母通常短于韵母。这里用 pinyin 估一个边界，避免把声母段拉得太长；
    零声母音节则把完整窗口交给韵母评分。
    """
    if initial == "零声母":
        return 0.0
    if initial in {"zh", "ch", "sh"}:
        base = 0.34
    elif initial in {"z", "c", "s", "j", "q", "x", "r"}:
        base = 0.30
    else:
        base = 0.26

    if syllable_duration < 0.16:
        return min(base, 0.22)
    return base


def _subsegment_cost(user_mfcc: np.ndarray,
                     ref_mfcc: np.ndarray,
                     user_start: float,
                     user_end: float,
                     ref_start: float,
                     ref_end: float) -> Optional[float]:
    """计算声母/韵母短片段距离；空片段返回 None。"""
    if user_end <= user_start or ref_end <= ref_start:
        return None
    u_slice = _slice_mfcc(user_mfcc, user_start, user_end)
    r_slice = _slice_mfcc(ref_mfcc, ref_start, ref_end)
    if u_slice.size == 0 or r_slice.size == 0:
        return None
    return _dtw_mean_cost(u_slice, r_slice)


# --------------------------------------------------------------- public API
def score_accuracy(user_mfcc: np.ndarray,
                   ref_mfcc: np.ndarray,
                   user_alignment: List,
                   ref_alignment: List,
                   user_voiced: Optional[np.ndarray] = None,
                   ref_voiced: Optional[np.ndarray] = None) -> AccuracyScore:
    """Compare user vs. reference MFCC per syllable using DTW.

    Args:
        user_mfcc:      [T_u, D] MFCC of the user's recording.
        ref_mfcc:       [T_r, D] MFCC of the reference (TTS) recording.
        user_alignment: list of SyllableAlign for the user signal.
        ref_alignment:  list of SyllableAlign for the reference signal,
                        same length and same chars as user_alignment.
    """
    if not user_alignment or not ref_alignment:
        return AccuracyScore(overall=0.0)

    n = min(len(user_alignment), len(ref_alignment))
    per: List[SyllableAccuracy] = []
    for i in range(n):
        ua = user_alignment[i]
        ra = ref_alignment[i]
        u_slice = _slice_mfcc(user_mfcc, ua.start, ua.end)
        r_slice = _slice_mfcc(ref_mfcc, ra.start, ra.end)
        cost = _dtw_mean_cost(u_slice, r_slice)
        dtw_score = _cost_to_score(cost)

        user_ratio = _voiced_ratio(user_voiced, ua.start, ua.end)
        ref_ratio = _voiced_ratio(ref_voiced, ra.start, ra.end)
        voiced_score = _voiced_coverage_score(user_ratio, ref_ratio)
        dur_score = _duration_score(ua.duration, ra.duration)
        score, articulation_score = _combine_accuracy(
            dtw_score, voiced_score, dur_score,
        )

        initial, final = _split_pinyin(getattr(ua, "pinyin", ""))
        ratio = _initial_ratio(initial, ua.duration)
        u_mid = ua.start + ua.duration * ratio
        r_mid = ra.start + ra.duration * ratio

        initial_cost = _subsegment_cost(
            user_mfcc, ref_mfcc, ua.start, u_mid, ra.start, r_mid,
        )
        final_cost = _subsegment_cost(
            user_mfcc, ref_mfcc, u_mid, ua.end, r_mid, ra.end,
        )
        initial_score = (
            _cost_to_score(initial_cost) if initial_cost is not None else None
        )
        final_score = _cost_to_score(final_cost) if final_cost is not None else None

        per.append(SyllableAccuracy(
            char=ua.char, score=score, dtw_cost=cost,
            duration=ua.duration,
            initial_score=initial_score,
            final_score=final_score,
            initial_cost=initial_cost,
            final_cost=final_cost,
            initial=initial,
            final=final,
            articulation_score=articulation_score,
            voiced_coverage_score=voiced_score,
            duration_score=dur_score,
            user_voiced_ratio=user_ratio,
            ref_voiced_ratio=ref_ratio,
        ))

    # Weighted average by syllable duration (longer syllables count more).
    if per:
        weights = np.array([max(s.duration, 0.05) for s in per])
        scores = np.array([s.score for s in per])
        overall = float(np.average(scores, weights=weights))
    else:
        overall = 0.0
    return AccuracyScore(overall=overall, per_syllable=per)
