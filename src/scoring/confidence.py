"""评分可信度估计。

这个模块不改变五维评分，只判断本次评分依据是否充分。它把 ASR 覆盖、
F0 有声比例、发声覆盖、有效语音时长和 TTS 参考状态合成一个 0..100
的可靠性分，供报告解释使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ConfidenceScore:
    overall: float
    dims: dict[str, float] = field(default_factory=dict)
    per_syllable: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _mean_present(values: list[float | None], default: float = 100.0) -> float:
    present = [float(v) for v in values if v is not None]
    if not present:
        return default
    return float(np.mean(present))


def _voiced_ratio(voiced_mask: np.ndarray | None) -> float:
    if voiced_mask is None or voiced_mask.size == 0:
        return 0.0
    return float(np.mean(voiced_mask))


def score_confidence(dim_scores: dict,
                     *,
                     user_voiced: np.ndarray | None = None,
                     user_duration: float = 0.0,
                     user_voiced_duration: float = 0.0,
                     has_tts_reference: bool = False,
                     has_asr: bool = False) -> ConfidenceScore:
    """根据已有评分结果估计本次评测的可靠性。"""
    completeness = dim_scores.get("completeness")
    accuracy = dim_scores.get("accuracy")
    tone = dim_scores.get("tone")

    completeness_score = (
        _clip_score(float(getattr(completeness, "coverage", 0.0)) * 100.0)
        if completeness and has_asr else 45.0
    )
    reference_score = 100.0 if has_tts_reference else 55.0

    duration_ratio = (
        user_voiced_duration / max(float(user_duration), 1e-6)
        if user_duration > 0 else 0.0
    )
    signal_score = _clip_score(100.0 * duration_ratio / 0.55)
    if user_voiced_duration < 0.4:
        signal_score = min(signal_score, 45.0)

    global_f0_score = _clip_score(100.0 * _voiced_ratio(user_voiced) / 0.45)

    tone_coverages = []
    if tone:
        tone_coverages = [
            getattr(s, "coverage", None)
            for s in getattr(tone, "per_syllable", [])
        ]
    tone_score = _clip_score(_mean_present(
        [None if v is None else float(v) * 100.0 for v in tone_coverages],
        default=global_f0_score,
    ))
    tone_score = min(tone_score, global_f0_score)

    articulation_values = []
    if accuracy:
        articulation_values = [
            getattr(s, "articulation_score", None)
            for s in getattr(accuracy, "per_syllable", [])
        ]
    accuracy_conf = _clip_score(
        0.55 * _mean_present(articulation_values, default=reference_score)
        + 0.45 * reference_score,
    )

    dims = {
        "signal": signal_score,
        "reference": reference_score,
        "asr": completeness_score,
        "f0": tone_score,
        "accuracy": accuracy_conf,
    }

    overall = (
        0.25 * dims["signal"]
        + 0.20 * dims["reference"]
        + 0.20 * dims["asr"]
        + 0.20 * dims["f0"]
        + 0.15 * dims["accuracy"]
    )

    per_syllable = _per_syllable_confidence(accuracy, tone, completeness)
    notes = _notes(dims, has_tts_reference, has_asr)
    return ConfidenceScore(
        overall=round(_clip_score(overall), 2),
        dims={k: round(_clip_score(v), 2) for k, v in dims.items()},
        per_syllable=per_syllable,
        notes=notes,
    )


def _per_syllable_confidence(accuracy, tone, completeness) -> list[dict[str, Any]]:
    if not accuracy or not tone:
        return []

    n = min(len(accuracy.per_syllable), len(tone.per_syllable))
    out: list[dict[str, Any]] = []
    for i in range(n):
        a = accuracy.per_syllable[i]
        t = tone.per_syllable[i]
        c = None
        if completeness and i < len(getattr(completeness, "per_syllable", [])):
            c = completeness.per_syllable[i]

        articulation = getattr(a, "articulation_score", None)
        tone_coverage = getattr(t, "coverage", None)
        completeness_ok = True if c is None else bool(getattr(c, "covered", True))
        pieces = [
            100.0 if completeness_ok else 35.0,
            _clip_score((tone_coverage or 0.0) * 100.0),
            100.0 if articulation is None else float(articulation),
        ]
        score = _clip_score(float(np.mean(pieces)))
        label = "高" if score >= 80 else "中" if score >= 60 else "低"
        out.append({
            "char": getattr(a, "char", ""),
            "score": round(score, 1),
            "level": label,
        })
    return out


def _notes(dims: dict[str, float],
           has_tts_reference: bool,
           has_asr: bool) -> list[str]:
    notes: list[str] = []
    if dims["signal"] < 60:
        notes.append("有效语音时长偏短，流利度和逐字分数需要谨慎看。")
    if dims["f0"] < 60:
        notes.append("可用 F0 帧偏少，声调分可靠性偏低。")
    if not has_tts_reference:
        notes.append("没有可用 TTS 标准音，准确度和韵律分采用弱参考。")
    if not has_asr:
        notes.append("没有可用 ASR 结果，完整度和漏读定位可靠性偏低。")
    if dims["asr"] < 70 and has_asr:
        notes.append("ASR 覆盖率偏低，可能存在漏读、错读或环境噪声。")
    return notes
