"""Aggregator (M4.5) — combine all dimension scores into one report.

The weights live in `config.DIM_WEIGHTS` so a TA can tweak the rubric
without touching code.  We also surface a flat per-syllable list that the
UI uses to draw the "char colour grid" feedback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from config import DIM_WEIGHTS


@dataclass
class ScoreResult:
    overall: float
    dims: Dict[str, float] = field(default_factory=dict)
    per_syllable: List[dict] = field(default_factory=list)
    fluency_detail: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def aggregate(dim_scores: dict) -> ScoreResult:
    """Combine sub-scores into a final report.

    Args:
        dim_scores: dict like {
            "accuracy":     AccuracyScore,
            "tone":         ToneScore,
            "fluency":      FluencyScore,
            "prosody":      ProsodyScore,
            "completeness": CompletenessScore,
        }
    """
    overall = 0.0
    total_w = 0.0
    dims = {}
    for name, weight in DIM_WEIGHTS.items():
        if name not in dim_scores:
            continue
        score = float(getattr(dim_scores[name], "overall", 0.0))
        dims[name] = score
        overall += weight * score
        total_w += weight
    overall = overall / total_w if total_w > 0 else 0.0

    # Build the per-syllable view (merge accuracy + tone by index).
    per_syll: List[dict] = []
    acc = dim_scores.get("accuracy")
    tone = dim_scores.get("tone")
    if acc and tone:
        n = min(len(acc.per_syllable), len(tone.per_syllable))
        for i in range(n):
            a = acc.per_syllable[i]
            t = tone.per_syllable[i]
            per_syll.append({
                "char":           a.char,
                "expected_tone":  t.expected,
                "detected_tone":  t.detected,
                "tone_ok":        bool(t.correct),
                "tone_score":     round(t.score, 1),
                "acc_score":      round(a.score, 1),
                "note":           "" if (t.correct and a.score >= 60)
                                  else _explain(a, t),
            })

    notes: List[str] = []
    if "completeness" in dim_scores:
        cov = dim_scores["completeness"].coverage
        if cov < 0.8:
            notes.append(f"识别覆盖率仅 {cov*100:.1f}%，可能漏读或环境噪声较大。")
    if "fluency" in dim_scores:
        fl = dim_scores["fluency"]
        if fl.pause_count > 0:
            notes.append(f"检测到 {fl.pause_count} 处明显停顿，语速 {fl.speech_rate:.2f} 字/秒。")

    fluency_detail = {}
    if "fluency" in dim_scores:
        fl = dim_scores["fluency"]
        fluency_detail = {
            "speech_rate":  fl.speech_rate,
            "pause_count":  fl.pause_count,
            "pause_total":  fl.pause_total,
            "pause_ratio":  fl.pause_ratio,
            **fl.detail,
        }

    return ScoreResult(
        overall=float(round(overall, 2)),
        dims={k: float(round(v, 2)) for k, v in dims.items()},
        per_syllable=per_syll,
        fluency_detail=fluency_detail,
        notes=notes,
    )


def _explain(acc, tone) -> str:
    bits = []
    if not tone.correct:
        if tone.detected == 0:
            bits.append(f"声调难判（应{tone.expected}声）")
        else:
            bits.append(f"声调误读 {tone.expected}→{tone.detected}")
    if acc.score < 60:
        bits.append("发音偏差")
    return "；".join(bits)
