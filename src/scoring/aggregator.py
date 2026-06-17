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
    confidence: dict = field(default_factory=dict)
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

    # Build the per-syllable view (merge accuracy + tone + completeness by index).
    per_syll: List[dict] = []
    acc = dim_scores.get("accuracy")
    tone = dim_scores.get("tone")
    completeness = dim_scores.get("completeness")
    confidence = dim_scores.get("confidence")
    if acc and tone:
        n = min(len(acc.per_syllable), len(tone.per_syllable))
        for i in range(n):
            a = acc.per_syllable[i]
            t = tone.per_syllable[i]
            c = None
            if completeness and i < len(getattr(completeness, "per_syllable", [])):
                c = completeness.per_syllable[i]
            lexical_tone = getattr(t, "lexical_tone", t.expected) or t.expected
            tone_rule = getattr(t, "tone_rule", "")
            completeness_ok = True if c is None else bool(getattr(c, "covered", True))
            syll_conf = None
            if confidence and i < len(getattr(confidence, "per_syllable", [])):
                syll_conf = confidence.per_syllable[i]
            per_syll.append({
                "char":           a.char,
                "pinyin":         getattr(t, "pinyin", ""),
                "lexical_tone":   lexical_tone,
                "expected_tone":  t.expected,
                "detected_tone":  t.detected,
                "tone_rule":      tone_rule,
                "tone_ok":        bool(t.correct),
                "tone_confidence": round(float(getattr(t, "confidence", 0.0)), 2),
                "tone_score":     round(t.score, 1),
                "ref_similarity": _round_optional(getattr(t, "ref_similarity", None), 1),
                "contour_score":  _round_optional(getattr(t, "contour_score", None), 1),
                "slope_score":    _round_optional(getattr(t, "slope_score", None), 1),
                "coverage":       _round_optional(getattr(t, "coverage", None), 2),
                "acc_score":      round(a.score, 1),
                "initial":        getattr(a, "initial", ""),
                "final":          getattr(a, "final", ""),
                "initial_score":  _round_optional(getattr(a, "initial_score", None), 1),
                "final_score":    _round_optional(getattr(a, "final_score", None), 1),
                "articulation_score": _round_optional(
                    getattr(a, "articulation_score", None), 1,
                ),
                "voiced_coverage_score": _round_optional(
                    getattr(a, "voiced_coverage_score", None), 1,
                ),
                "duration_score":  _round_optional(getattr(a, "duration_score", None), 1),
                "user_voiced_ratio": _round_optional(
                    getattr(a, "user_voiced_ratio", None), 2,
                ),
                "completeness_ok": completeness_ok,
                "confidence_score": (
                    None if syll_conf is None else syll_conf.get("score")
                ),
                "confidence_level": (
                    "" if syll_conf is None else syll_conf.get("level", "")
                ),
                "start":          round(float(getattr(t, "start", 0.0)), 3),
                "end":            round(float(getattr(t, "end", 0.0)), 3),
                "note":           "" if (t.correct and a.score >= 60 and completeness_ok)
                                  else _explain(a, t, completeness_ok),
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
    if "confidence" in dim_scores:
        notes.extend(getattr(dim_scores["confidence"], "notes", []))

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
        confidence=_confidence_dict(dim_scores.get("confidence")),
        notes=notes,
    )


def _explain(acc, tone, completeness_ok: bool = True) -> str:
    bits = []
    if not completeness_ok:
        bits.append("疑似漏读")
    if not tone.correct:
        if tone.detected == 0:
            bits.append(f"声调难判（应{tone.expected}声）")
        elif getattr(tone, "reason", ""):
            bits.append(getattr(tone, "reason"))
        else:
            bits.append(f"声调误读 {tone.expected}→{tone.detected}")
    if acc.score < 60:
        bits.append("发音偏差")
    initial_score = getattr(acc, "initial_score", None)
    final_score = getattr(acc, "final_score", None)
    if initial_score is not None and initial_score < 60:
        bits.append("声母偏差")
    if final_score is not None and final_score < 60:
        bits.append("韵母偏差")
    articulation_score = getattr(acc, "articulation_score", None)
    if articulation_score is not None and articulation_score < 60:
        bits.append("有效发声不足")
    if getattr(tone, "tone_rule", ""):
        bits.append(f"参考声调按“{tone.tone_rule}”处理")
    return "；".join(bits)


def _round_optional(value, digits: int):
    if value is None:
        return None
    return round(float(value), digits)


def _confidence_dict(confidence) -> dict:
    if not confidence:
        return {}
    return {
        "overall": round(float(getattr(confidence, "overall", 0.0)), 2),
        "dims": getattr(confidence, "dims", {}),
        "notes": getattr(confidence, "notes", []),
    }
