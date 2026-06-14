"""Pronunciation accuracy via DTW (M4.1).

For each aligned syllable we cut the matching MFCC window out of both the
**user** and the **reference (TTS)** signals, then measure how far apart the
two MFCC sequences are using Dynamic Time Warping.  Smaller DTW cost → closer
to the reference → higher score.

This is the "Plan A" route in the design doc (light-weight, no GOP needed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import librosa
import numpy as np

from config import HOP_LENGTH, SAMPLE_RATE


@dataclass
class SyllableAccuracy:
    char: str
    score: float            # 0..100
    dtw_cost: float         # raw mean-cost from DTW
    duration: float         # syllable duration in seconds


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


# --------------------------------------------------------------- public API
def score_accuracy(user_mfcc: np.ndarray,
                   ref_mfcc: np.ndarray,
                   user_alignment: List,
                   ref_alignment: List) -> AccuracyScore:
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
        score = _cost_to_score(cost)
        per.append(SyllableAccuracy(
            char=ua.char, score=score, dtw_cost=cost,
            duration=ua.duration,
        ))

    # Weighted average by syllable duration (longer syllables count more).
    if per:
        weights = np.array([max(s.duration, 0.05) for s in per])
        scores = np.array([s.score for s in per])
        overall = float(np.average(scores, weights=weights))
    else:
        overall = 0.0
    return AccuracyScore(overall=overall, per_syllable=per)
