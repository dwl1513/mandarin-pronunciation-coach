"""完整度评分：判断用户是否读完整个参考文本。

整体分数使用字符级 CER；逐字覆盖状态使用参考文本和识别文本的最长公共
子序列。这样既保留整句完整度，也能在报告里标出疑似漏读的字。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from jiwer import cer


@dataclass
class SyllableCompleteness:
    char: str
    covered: bool
    hyp_index: int | None = None


@dataclass
class CompletenessScore:
    overall: float       # 0..100
    cer: float           # raw character error rate
    coverage: float      # 1 - CER, clipped
    per_syllable: list[SyllableCompleteness] = field(default_factory=list)


_DIGIT_TRANSLATION = str.maketrans({
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
    "〇": "零",
    "○": "零",
    "Ｏ": "零",
    "O": "零",
    "o": "零",
})


def _normalized_chars(s: str) -> str:
    """保留中文并统一数字写法，避免“二000/二零零零”被误判为漏读。"""
    normalized = s.translate(_DIGIT_TRANSLATION)
    return "".join(ch for ch in normalized if "一" <= ch <= "鿿")


def _lcs_covered(ref: str, hyp: str) -> list[SyllableCompleteness]:
    """用最长公共子序列给参考文本逐字标记是否被识别覆盖。"""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if ref[i] == hyp[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    covered: list[SyllableCompleteness] = []
    i = j = 0
    while i < n:
        if j < m and ref[i] == hyp[j]:
            covered.append(SyllableCompleteness(ref[i], True, j))
            i += 1
            j += 1
        elif j < m and dp[i][j + 1] >= dp[i + 1][j]:
            j += 1
        else:
            covered.append(SyllableCompleteness(ref[i], False, None))
            i += 1
    return covered


def score_completeness(recognized: str, reference: str) -> CompletenessScore:
    ref = _normalized_chars(reference)
    hyp = _normalized_chars(recognized)
    if not ref:
        return CompletenessScore(0.0, 1.0, 0.0)
    if not hyp:
        per = [SyllableCompleteness(ch, False, None) for ch in ref]
        return CompletenessScore(0.0, 1.0, 0.0, per)
    err = float(cer(ref, hyp))
    coverage = max(0.0, 1.0 - err)
    return CompletenessScore(
        overall=float(min(100.0, coverage * 100.0)),
        cer=err,
        coverage=coverage,
        per_syllable=_lcs_covered(ref, hyp),
    )
