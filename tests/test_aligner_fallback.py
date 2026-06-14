"""Verify the VAD-based fallback in src.asr.aligner.

We do NOT load wav2vec2 here — the goal is to ensure that a bare-minimum
alignment is still produced when the model is unavailable, so downstream
scoring keeps running.
"""
import numpy as np

from src.asr.aligner import _parse_reference, _vad_uniform_align


def test_parse_reference_extracts_chinese_only():
    parsed = _parse_reference("你好, world! 朋友。")
    chars = [c for c, _p, _t in parsed]
    assert chars == ["你", "好", "朋", "友"]
    tones = [t for _c, _p, t in parsed]
    # 你=3, 好=3, 朋=2, 友=3 (pypinyin defaults)
    assert tones[0] == 3 and tones[1] == 3


def test_parse_reference_empty_for_punctuation_only():
    assert _parse_reference("?.,!") == []


def test_vad_uniform_align_covers_all_chars():
    chars = _parse_reference("你好朋友")
    vad = [(0.5, 2.5)]                          # 2 s of voiced speech
    out = _vad_uniform_align(chars, vad, total_duration=3.0)
    assert len(out) == 4
    # All start times within voiced region, end > start, monotone increasing.
    last = 0.0
    for syl in out:
        assert 0.5 <= syl.start <= 2.5
        assert syl.end > syl.start
        assert syl.start >= last - 1e-3
        last = syl.start
    # All marked as not in vocab (fallback path)
    assert all(not s.in_vocab for s in out)


def test_vad_uniform_align_with_no_segments_uses_full_duration():
    chars = _parse_reference("你好")
    out = _vad_uniform_align(chars, [], total_duration=1.0)
    assert len(out) == 2
    assert out[-1].end <= 1.01
