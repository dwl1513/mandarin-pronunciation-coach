"""ASR via wav2vec2 + CTC greedy decode.

Used for:
    1. completeness scoring  — how much of the reference did the user actually
                                read (character-level edit similarity).
    2. sanity-check          — if recognition is wildly off, we down-weight the
                                forced-alignment result.
"""
from __future__ import annotations

import numpy as np

from .models import get_log_probs


def recognize(wav: np.ndarray) -> str:
    """Run CTC greedy decode on a 16 kHz mono waveform → recognized text.

    The vocab of `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`
    occasionally contains non-Chinese tokens (pinyin / English bigrams). On
    out-of-distribution audio (TTS, very clean studio recordings) the model
    sometimes emits these tokens between Chinese chars, e.g.
    `'今 TI天 B步 B吧'`.  We strip everything that isn't a CJK character or
    whitespace so completeness scoring and the report stay readable.
    """
    log_probs, bundle = get_log_probs(wav)
    pred_ids = log_probs.argmax(dim=-1)[0].cpu().tolist()
    raw = bundle.processor.decode(pred_ids).strip()
    cleaned = "".join(
        ch for ch in raw
        if ("一" <= ch <= "鿿") or ch.isspace()
    )
    return " ".join(cleaned.split())
