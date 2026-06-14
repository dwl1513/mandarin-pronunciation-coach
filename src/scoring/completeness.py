"""Completeness scoring — did the user read the whole reference text?

Uses character-level edit distance between recognized text and reference.
Only the longest matching subsequence counts, so swapping word order or
adding filler still costs the speaker.
"""
from __future__ import annotations

from dataclasses import dataclass

from jiwer import cer


@dataclass
class CompletenessScore:
    overall: float       # 0..100
    cer: float           # raw character error rate
    coverage: float      # 1 - CER, clipped


def _chinese_only(s: str) -> str:
    return "".join(ch for ch in s if "一" <= ch <= "鿿")


def score_completeness(recognized: str, reference: str) -> CompletenessScore:
    ref = _chinese_only(reference)
    hyp = _chinese_only(recognized)
    if not ref:
        return CompletenessScore(0.0, 1.0, 0.0)
    if not hyp:
        return CompletenessScore(0.0, 1.0, 0.0)
    err = float(cer(ref, hyp))
    coverage = max(0.0, 1.0 - err)
    return CompletenessScore(
        overall=float(min(100.0, coverage * 100.0)),
        cer=err,
        coverage=coverage,
    )
