"""Forced alignment (M2).

Given an audio waveform and the *reference* Chinese text the speaker should
have read, produce a per-character time axis:

    [(char, pinyin, tone, start_s, end_s, in_vocab), ...]

Primary path:
    1. Run the same wav2vec2 CTC model used by `recognize()` to get log-probs
       at ~50 Hz.
    2. Tokenize the reference text into the model's character vocabulary.
    3. Call torchaudio.functional.forced_align to obtain a frame-level
       Viterbi alignment.
    4. Walk through the alignment, collapsing consecutive same-token frames
       into one span per reference character.

Fallback path (model load fails, too many OOV chars, very short audio):
    Take the voiced regions produced by VAD and distribute the reference
    characters uniformly across the total voiced time.  Coarse, but lets
    downstream scoring keep running even when the model isn't available.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
from pypinyin import Style, lazy_pinyin

from config import SAMPLE_RATE


@dataclass
class SyllableAlign:
    char: str               # the Chinese character
    pinyin: str             # base pinyin (no tone diacritic)
    tone: int               # expected tone 1..4, 5 = neutral
    start: float            # seconds
    end: float              # seconds
    in_vocab: bool = True   # True if the char existed in the ASR vocab
    lexical_tone: int = 0   # dictionary tone before tone sandhi
    tone_rule: str = ""     # applied sandhi rule, empty when unchanged

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParsedSyllable:
    char: str
    pinyin: str
    tone: int
    lexical_tone: int
    tone_rule: str = ""


# ---------------------------------------------------------------- text helpers
def _parse_reference_detail(text: str) -> List[ParsedSyllable]:
    """Strip punctuation and return pinyin with common Mandarin tone sandhi.

    `pypinyin` returns dictionary tones.  In real connected Mandarin, common
    words like "你好"、"不是"、"一个" are read with changed tones.  Using the
    dictionary tone directly makes the per-character report mark many correct
    readings as tone errors, so we apply the high-frequency PSC-relevant rules
    here.
    """
    keep: List[str] = []
    for ch in text:
        if "一" <= ch <= "鿿":         # CJK unified ideographs
            keep.append(ch)
    if not keep:
        return []

    # pypinyin returns e.g. "ni3"; numeric style keeps the tone digit.
    with_tone = lazy_pinyin(keep, style=Style.TONE3, neutral_tone_with_five=True)
    plain     = lazy_pinyin(keep, style=Style.NORMAL)

    out: List[ParsedSyllable] = []
    for ch, py_tone, py_plain in zip(keep, with_tone, plain):
        # Extract trailing digit as tone; default to 5 (neutral) if absent.
        tone = 5
        if py_tone and py_tone[-1].isdigit():
            tone = int(py_tone[-1])
        out.append(ParsedSyllable(ch, py_plain, tone, tone))
    _apply_tone_sandhi(out)
    return out


def _parse_reference(text: str) -> List[Tuple[str, str, int]]:
    """Compatibility helper used by tests and fallback callers."""
    return [(s.char, s.pinyin, s.tone) for s in _parse_reference_detail(text)]


def _set_sandhi(syl: ParsedSyllable, tone: int, rule: str) -> None:
    if syl.tone == tone and syl.tone_rule:
        return
    syl.tone = tone
    syl.tone_rule = rule


def _apply_tone_sandhi(syllables: List[ParsedSyllable]) -> None:
    """Apply common tone-sandhi rules used in ordinary Mandarin reading."""
    n = len(syllables)
    if n == 0:
        return

    # 三声连读：连续三声中，除最后一个外通常改读二声。
    i = 0
    while i < n:
        if syllables[i].lexical_tone != 3:
            i += 1
            continue
        j = i
        while j < n and syllables[j].lexical_tone == 3:
            j += 1
        if j - i >= 2:
            for k in range(i, j - 1):
                _set_sandhi(syllables[k], 2, "三声连读")
        i = j

    for i, syl in enumerate(syllables):
        next_syl = syllables[i + 1] if i + 1 < n else None
        prev_syl = syllables[i - 1] if i > 0 else None

        if syl.char == "不":
            if next_syl and next_syl.lexical_tone == 4:
                _set_sandhi(syl, 2, "不 + 四声")
            else:
                _set_sandhi(syl, 4, "不字本调")

        if syl.char == "一":
            if prev_syl and next_syl and prev_syl.char == next_syl.char:
                _set_sandhi(syl, 5, "重叠动词中间的一")
            elif next_syl and next_syl.lexical_tone == 4:
                _set_sandhi(syl, 2, "一 + 四声")
            elif next_syl:
                _set_sandhi(syl, 4, "一 + 非四声")
            else:
                _set_sandhi(syl, 1, "一字单读")


def _fields(syl) -> Tuple[str, str, int, int, str]:
    """Accept both ParsedSyllable and legacy 3-tuples in fallback tests."""
    if isinstance(syl, ParsedSyllable):
        return syl.char, syl.pinyin, syl.tone, syl.lexical_tone, syl.tone_rule
    ch, py, tone = syl
    return ch, py, tone, tone, ""


# --------------------------------------------------------------- fallback path
def _vad_uniform_align(chars: List,
                       vad_segments: List[Tuple[float, float]],
                       total_duration: float) -> List[SyllableAlign]:
    """Distribute characters evenly across all voiced time."""
    if not chars:
        return []

    if vad_segments:
        # Concatenate voiced regions; allot duration proportional to char count.
        durations = [end - start for start, end in vad_segments]
        total_voiced = sum(durations)
        if total_voiced <= 0:
            vad_segments = [(0.0, total_duration)]
            durations = [total_duration]
            total_voiced = total_duration
    else:
        vad_segments = [(0.0, total_duration)]
        durations = [total_duration]
        total_voiced = total_duration

    per_char = total_voiced / len(chars)
    aligned: List[SyllableAlign] = []
    seg_idx = 0
    seg_start, seg_end = vad_segments[seg_idx]
    cursor = seg_start
    for syl in chars:
        ch, py, tone, lexical_tone, tone_rule = _fields(syl)
        # If current segment runs out, advance to the next voiced block.
        remaining = seg_end - cursor
        if remaining < per_char * 0.5 and seg_idx + 1 < len(vad_segments):
            seg_idx += 1
            seg_start, seg_end = vad_segments[seg_idx]
            cursor = seg_start
        start = cursor
        end = min(cursor + per_char, seg_end)
        aligned.append(SyllableAlign(
            char=ch, pinyin=py, tone=tone,
            start=float(start), end=float(end), in_vocab=False,
            lexical_tone=lexical_tone, tone_rule=tone_rule,
        ))
        cursor = end
    return aligned


# --------------------------------------------------------------- primary path
def _spans_from_alignment(aligned_ids: list, blank_id: int,
                          n_expected: int) -> Optional[List[Tuple[int, int]]]:
    """Collapse the CTC alignment path into one (start_frame, end_frame) per char.

    CTC tends to *peak-align* tokens — each character only "fires" for 1-3
    frames while the rest is blank.  Tone scoring needs the full voiced vowel
    inside the window, so we widen each span to share blank gaps with its
    neighbours (midpoint split).

    Returns None when the number of distinct runs does not match the target
    length — that signals a degenerate alignment we shouldn't trust.
    """
    raw: List[Tuple[int, int]] = []
    n = len(aligned_ids)
    t = 0
    while t < n:
        if aligned_ids[t] == blank_id:
            t += 1
            continue
        start = t
        tok = aligned_ids[t]
        while t < n and aligned_ids[t] == tok:
            t += 1
        raw.append((start, t))
    if len(raw) != n_expected:
        return None
    if not raw:
        return raw

    # Extend each peak-aligned span outward so tone scoring has enough voiced
    # vowel frames — BUT cap the extension so we don't bleed into neighbouring
    # syllables.  At wav2vec2's 50 Hz frame rate, 4 frames ≈ 80 ms, which is
    # roughly the longest Mandarin vowel half-duration we want to absorb.
    # Without this cap a long inter-syllable silence (~200 ms blank) would
    # push each window all the way to the next syllable's start, capturing the
    # tail of the previous vowel — which inverts tone 2 (rising) into a
    # falling shape and similarly garbles every other tone.
    MAX_EXTEND = 4
    expanded: List[Tuple[int, int]] = []
    for i, (s, e) in enumerate(raw):
        if i == 0:
            left = s
        else:
            midpoint = (raw[i - 1][1] + s) // 2
            left = max(midpoint, s - MAX_EXTEND)
        if i == len(raw) - 1:
            right = e
        else:
            midpoint = (e + raw[i + 1][0]) // 2
            right = min(midpoint, e + MAX_EXTEND)
        expanded.append((left, max(right, left + 1)))
    return expanded


def _wav2vec2_align(wav: np.ndarray,
                    chars: List[ParsedSyllable]
                    ) -> Optional[List[SyllableAlign]]:
    """Try the model-based path. Return None on any failure."""
    try:
        import torch
        import torchaudio.functional as taF
        from .models import get_log_probs
    except Exception:
        return None

    try:
        log_probs, bundle = get_log_probs(wav)              # [1, T, V]
    except Exception:
        return None

    vocab = bundle.vocab()
    blank_id = bundle.blank_id

    # Build target id sequence; abort if too many OOV characters.
    target_ids: List[int] = []
    oov_count = 0
    for syl in chars:
        ch = syl.char
        if ch in vocab:
            target_ids.append(int(vocab[ch]))
        else:
            oov_count += 1
            target_ids.append(-1)
    if oov_count > 0 and oov_count >= max(1, len(chars) // 3):
        return None             # >=1/3 OOV → trust the fallback instead

    # Skip OOV positions: align only the in-vocab subset, then re-insert OOVs
    # later with interpolated timestamps.
    keep = [(i, tid) for i, tid in enumerate(target_ids) if tid >= 0]
    if not keep:
        return None
    kept_ids = [tid for _, tid in keep]

    targets = torch.tensor([kept_ids], dtype=torch.int32, device=bundle.device)
    input_lengths = torch.tensor([log_probs.shape[1]], dtype=torch.int32,
                                  device=bundle.device)
    target_lengths = torch.tensor([len(kept_ids)], dtype=torch.int32,
                                   device=bundle.device)
    try:
        aligned, _scores = taF.forced_align(
            log_probs.to(torch.float32),
            targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=blank_id,
        )
    except Exception:
        return None

    aligned_ids = aligned[0].cpu().tolist()
    spans = _spans_from_alignment(aligned_ids, blank_id, len(kept_ids))
    if spans is None:
        return None

    frame_sec = 1.0 / bundle.frame_rate_hz
    # Map kept-position → (start_s, end_s)
    kept_intervals: dict = {}
    for (orig_idx, _tid), (s_frame, e_frame) in zip(keep, spans):
        kept_intervals[orig_idx] = (s_frame * frame_sec, e_frame * frame_sec)

    # Interpolate any OOV gaps so every char has a timestamp.
    result: List[SyllableAlign] = []
    last_end = 0.0
    next_known = sorted(kept_intervals.keys())
    next_ptr = 0
    for i, syl in enumerate(chars):
        if i in kept_intervals:
            s, e = kept_intervals[i]
            result.append(SyllableAlign(
                syl.char, syl.pinyin, syl.tone, float(s), float(e), True,
                lexical_tone=syl.lexical_tone, tone_rule=syl.tone_rule,
            ))
            last_end = e
            while next_ptr < len(next_known) and next_known[next_ptr] <= i:
                next_ptr += 1
        else:
            # find next known anchor; spread OOV chars evenly in between
            next_anchor_idx = next_known[next_ptr] if next_ptr < len(next_known) else None
            if next_anchor_idx is not None:
                gap_start = last_end
                gap_end = kept_intervals[next_anchor_idx][0]
                # simpler: count OOV chars between last anchor and next anchor
                slots = sum(1 for j in range(i, next_anchor_idx)
                            if j not in kept_intervals)
                slots = max(slots, 1)
                slice_dur = (gap_end - gap_start) / slots
                start = gap_start + (i - max(
                    [k for k in kept_intervals if k < i] + [-1])
                                     - 1) * slice_dur
                end = start + slice_dur
            else:
                # OOV chars trailing the last anchor — append at the tail
                start = last_end
                end = last_end + 0.15
                last_end = end
            result.append(SyllableAlign(
                syl.char, syl.pinyin, syl.tone, float(start), float(end), False,
                lexical_tone=syl.lexical_tone, tone_rule=syl.tone_rule,
            ))
    return result


# ----------------------------------------------------------------- public API
def align(wav: np.ndarray,
          reference_text: str,
          vad_segments: Optional[List[Tuple[float, float]]] = None,
          sr: int = SAMPLE_RATE) -> List[SyllableAlign]:
    """Align audio to reference text → per-character timestamps.

    Falls back to a VAD-based uniform partition if the model is unavailable.
    """
    chars = _parse_reference_detail(reference_text)
    if not chars or wav.size == 0:
        return []

    result = _wav2vec2_align(wav, chars)
    if result is not None and len(result) == len(chars):
        return result

    total_duration = len(wav) / float(sr)
    return _vad_uniform_align(chars, vad_segments or [], total_duration)
