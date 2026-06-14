"""Fluency scoring (M4.3).

Maps VAD-derived statistics — speech rate, pause count, pause ratio — onto
the PSC "流畅程度" criterion.  We aim for a defensible 0..100 number rather
than a single magic threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from config import MAX_PAUSE_RATIO, TARGET_SYLLABLES_PER_SEC


@dataclass
class FluencyScore:
    overall: float
    speech_rate: float        # syllables / second of voiced speech
    pause_count: int
    pause_total: float        # total silent time inside the utterance, sec
    pause_ratio: float        # pause_total / total_duration
    detail: dict = field(default_factory=dict)


def _internal_pauses(vad_segments: List[Tuple[float, float]]
                      ) -> Tuple[int, float]:
    """Count silences *between* voiced regions and their total duration."""
    if len(vad_segments) < 2:
        return 0, 0.0
    pauses = [
        s2 - e1
        for (s1, e1), (s2, e2) in zip(vad_segments[:-1], vad_segments[1:])
        if s2 > e1
    ]
    # Treat sub-200 ms gaps as natural co-articulation, not "pauses".
    real = [p for p in pauses if p > 0.2]
    return len(real), float(sum(real))


def score_fluency(vad_segments: List[Tuple[float, float]],
                  alignment: List,
                  total_duration: float) -> FluencyScore:
    """Combine speech rate and pause behaviour into one 0..100 score."""
    n_syllables = len(alignment)
    voiced_duration = sum(e - s for s, e in vad_segments)
    if voiced_duration <= 0.05 or total_duration <= 0.05:
        return FluencyScore(0.0, 0.0, 0, 0.0, 1.0, detail={"reason": "no voice"})

    speech_rate = n_syllables / voiced_duration
    pause_count, pause_total = _internal_pauses(vad_segments)
    pause_ratio = pause_total / max(total_duration, 1e-3)

    # Speech-rate score: peaks at TARGET_SYLLABLES_PER_SEC, decays both sides.
    rate_score = 100.0 * np.exp(
        -((speech_rate - TARGET_SYLLABLES_PER_SEC) ** 2) / (1.5 ** 2)
    )
    # Pause-ratio score: 100 when ratio==0, 0 when ratio>=MAX_PAUSE_RATIO.
    pause_score = float(np.clip(
        100.0 * (1.0 - pause_ratio / MAX_PAUSE_RATIO), 0.0, 100.0,
    ))
    # Lots of pauses (>5 in a 60-char passage feels choppy) is its own penalty.
    norm_pauses = pause_count / max(n_syllables / 8.0, 1.0)
    pause_count_score = float(np.clip(100.0 - 20.0 * max(0.0, norm_pauses - 1.0),
                                       0.0, 100.0))

    overall = 0.5 * rate_score + 0.3 * pause_score + 0.2 * pause_count_score
    return FluencyScore(
        overall=float(overall),
        speech_rate=float(speech_rate),
        pause_count=int(pause_count),
        pause_total=float(pause_total),
        pause_ratio=float(pause_ratio),
        detail={
            "rate_score": float(rate_score),
            "pause_score": float(pause_score),
            "pause_count_score": float(pause_count_score),
            "voiced_duration": float(voiced_duration),
        },
    )
