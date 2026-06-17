"""Verify the VAD-based fallback in src.asr.aligner.

We do NOT load wav2vec2 here — the goal is to ensure that a bare-minimum
alignment is still produced when the model is unavailable, so downstream
scoring keeps running.
"""

from src.asr.aligner import _parse_reference, _parse_reference_detail, _vad_uniform_align


def test_parse_reference_extracts_chinese_only():
    parsed = _parse_reference("你好, world! 朋友。")
    chars = [c for c, _p, _t in parsed]
    assert chars == ["你", "好", "朋", "友"]
    tones = [t for _c, _p, t in parsed]
    # "你好" 连续三声，前一个三声按普通话变调读二声。
    assert tones[0] == 2 and tones[1] == 3


def test_parse_reference_empty_for_punctuation_only():
    assert _parse_reference("?.,!") == []


def test_parse_reference_applies_third_tone_sandhi():
    parsed = _parse_reference_detail("你好")
    assert [(s.char, s.lexical_tone, s.tone, s.tone_rule) for s in parsed] == [
        ("你", 3, 2, "三声连读"),
        ("好", 3, 3, ""),
    ]


def test_parse_reference_applies_bu_and_yi_sandhi():
    parsed = _parse_reference_detail("不是一个看一看")
    tones = {s.char + str(i): (s.lexical_tone, s.tone, s.tone_rule)
             for i, s in enumerate(parsed)}
    assert tones["不0"] == (4, 2, "不 + 四声")
    assert tones["一2"] == (1, 2, "一 + 四声")
    assert tones["一5"] == (1, 5, "重叠动词中间的一")


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
