"""Prosody scoring (M4.4) — sentence-level naturalness.

We compare the overall F0 trajectory of the user against the reference
recording's trajectory.  Two complementary numbers:

    * DTW similarity of the (z-scored) F0 contours  — does the melody match?
    * Range ratio                                    — is the speaker
      flat-toning or over-acting?
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class ProsodyScore:
    overall: float
    contour_similarity: float        # 0..100
    range_score: float               # 0..100
    pitch_range_semitones: float     # raw measurement


def _voiced_only(f0: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    return f0[voiced & (f0 > 0)]


def _zscore(seq: np.ndarray) -> np.ndarray:
    if seq.size < 2:
        return seq
    mu = seq.mean()
    sd = seq.std()
    if sd < 1e-3:
        return np.zeros_like(seq)
    return (seq - mu) / sd


def score_prosody(user_f0: np.ndarray, user_voiced: np.ndarray,
                  ref_f0: np.ndarray, ref_voiced: np.ndarray) -> ProsodyScore:
    u = _voiced_only(user_f0, user_voiced)
    r = _voiced_only(ref_f0, ref_voiced)
    if u.size < 5 or r.size < 5:
        return ProsodyScore(0.0, 0.0, 0.0, 0.0)

    # Convert to semitones (re. 100 Hz), then z-score for shape comparison.
    u_st = 12.0 * np.log2(np.maximum(u, 1e-3) / 100.0)
    r_st = 12.0 * np.log2(np.maximum(r, 1e-3) / 100.0)
    u_z = _zscore(u_st)
    r_z = _zscore(r_st)

    D, _wp = librosa.sequence.dtw(X=u_z.reshape(1, -1), Y=r_z.reshape(1, -1),
                                   metric="euclidean")
    avg_cost = float(D[-1, -1]) / max(u_z.size + r_z.size, 1)
    # avg_cost ~0 => identical.  Connected speech between real users and TTS
    # naturally differs, so the reference curve is a guide rather than a hard
    # template.
    similarity = float(np.clip(100.0 * (1.0 - avg_cost / 3.0), 0.0, 100.0))

    # Pitch range: native expressive speech ~8-14 ST. <4 ST is flat reading.
    pitch_range = float(np.percentile(u_st, 95) - np.percentile(u_st, 5))
    range_score = float(np.clip(100.0 * pitch_range / 10.0, 0.0, 100.0))

    overall = 0.55 * similarity + 0.45 * range_score
    return ProsodyScore(
        overall=float(overall),
        contour_similarity=similarity,
        range_score=range_score,
        pitch_range_semitones=pitch_range,
    )
