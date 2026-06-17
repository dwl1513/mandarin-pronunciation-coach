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


def _speech_rate_score(speech_rate: float) -> float:
    """普通话朗读允许稳一点；2.5~5.0 字/秒都视为自然区间。"""
    if 2.5 <= speech_rate <= 5.0:
        return 100.0
    if speech_rate < 2.5:
        return float(np.clip(100.0 * (speech_rate / 2.5) ** 1.3, 0.0, 100.0))
    return float(np.clip(100.0 * np.exp(-((speech_rate - 5.0) / 2.0) ** 2),
                         0.0, 100.0))


def _pause_score(pause_total: float, pause_count: int,
                 total_duration: float, n_syllables: int) -> tuple[float, float]:
    """短句允许少量自然停顿；长停顿和高停顿占比才明显扣分。"""
    pause_ratio = pause_total / max(total_duration, 1e-3)
    if n_syllables >= 80:
        # PSC 朗读篇目约 400~600 字，正式范读会按语义分层停顿。这里按停顿
        # 占比和“每十字停顿次数”评分，避免把正常断句当作卡顿。
        ratio_target = 0.18
        ratio_limit = 0.38
        if pause_ratio <= ratio_target:
            ratio_score = 100.0
        else:
            ratio_score = float(np.clip(
                100.0 * (1.0 - (pause_ratio - ratio_target)
                         / (ratio_limit - ratio_target) * 0.45),
                55.0, 100.0,
            ))

        pauses_per_10 = pause_count / max(n_syllables / 10.0, 1e-3)
        if pauses_per_10 <= 0.95:
            count_score = 100.0
        else:
            count_score = float(np.clip(
                100.0 - 18.0 * (pauses_per_10 - 0.95), 70.0, 100.0,
            ))
        return pause_ratio, 0.72 * ratio_score + 0.28 * count_score

    # 短句里一个 0.3~0.5s 的停顿常常是 TTS/示范朗读的自然断句。
    free_pause = min(0.45, 0.06 * max(n_syllables, 1))
    adjusted_pause = max(0.0, pause_total - free_pause)
    adjusted_ratio = adjusted_pause / max(total_duration, 1e-3)
    ratio_score = float(np.clip(
        100.0 * (1.0 - adjusted_ratio / MAX_PAUSE_RATIO), 0.0, 100.0,
    ))

    allowed_pauses = max(1.0, n_syllables / 8.0)
    extra_pauses = max(0.0, pause_count - allowed_pauses)
    count_score = float(np.clip(100.0 - 18.0 * extra_pauses, 0.0, 100.0))
    return pause_ratio, 0.7 * ratio_score + 0.3 * count_score


def _rhythm_score(alignment: List) -> float:
    """用字间时长变异度衡量卡顿；对齐不可用时保持中性偏高。"""
    durations = np.asarray([
        max(0.0, float(getattr(s, "end", 0.0)) - float(getattr(s, "start", 0.0)))
        for s in alignment
    ], dtype=np.float32)
    durations = durations[durations > 0.03]
    if durations.size < 3:
        return 90.0
    median = float(np.median(durations))
    if median <= 1e-3:
        return 70.0
    cv = float(np.std(durations) / median)
    return float(np.clip(100.0 * (1.0 - max(0.0, cv - 0.45) / 0.75), 0.0, 100.0))


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
    pause_ratio, pause_score = _pause_score(
        pause_total, pause_count, total_duration, n_syllables,
    )

    rate_score = _speech_rate_score(speech_rate)
    rhythm_score = _rhythm_score(alignment)

    overall = 0.35 * rate_score + 0.40 * pause_score + 0.25 * rhythm_score
    if pause_ratio > MAX_PAUSE_RATIO:
        # 长时间沉默是流利度硬伤，不能被稳定字间节奏抵消。
        overall *= float(np.clip(1.0 - (pause_ratio - MAX_PAUSE_RATIO), 0.35, 1.0))
    return FluencyScore(
        overall=float(overall),
        speech_rate=float(speech_rate),
        pause_count=int(pause_count),
        pause_total=float(pause_total),
        pause_ratio=float(pause_ratio),
        detail={
            "rate_score": float(rate_score),
            "pause_score": float(pause_score),
            "rhythm_score": float(rhythm_score),
            "voiced_duration": float(voiced_duration),
            "target_syllables_per_sec": float(TARGET_SYLLABLES_PER_SEC),
        },
    )
