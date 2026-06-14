"""Unit tests for the scoring modules (M4)."""
from dataclasses import dataclass

import numpy as np

from src.asr.aligner import SyllableAlign
from src.scoring.accuracy import score_accuracy
from src.scoring.aggregator import aggregate
from src.scoring.completeness import score_completeness
from src.scoring.fluency import score_fluency
from src.scoring.prosody import score_prosody
from src.scoring.tone import score_tone


# --------------------------------------------------------- tone classification
def _fake_f0_segment(contour, hop_sec=0.01):
    """Build a (f0_array, voiced_mask, alignment) pinned to 0..n_frames*hop."""
    n = len(contour)
    f0 = np.asarray(contour, dtype=np.float32)
    voiced = f0 > 0
    return f0, voiced


def test_tone_flat_is_classified_as_tone1(f0_contour_factory):
    f0 = f0_contour_factory("flat")
    voiced = np.ones_like(f0, dtype=bool)
    align = [SyllableAlign(char="妈", pinyin="ma", tone=1,
                            start=0.0, end=len(f0) * 0.01)]
    score = score_tone(f0, voiced, align)
    assert score.per_syllable[0].detected == 1
    assert score.per_syllable[0].correct


def test_tone_rising_is_classified_as_tone2(f0_contour_factory):
    f0 = f0_contour_factory("rising")
    voiced = np.ones_like(f0, dtype=bool)
    align = [SyllableAlign(char="麻", pinyin="ma", tone=2,
                            start=0.0, end=len(f0) * 0.01)]
    score = score_tone(f0, voiced, align)
    assert score.per_syllable[0].detected == 2


def test_tone_falling_is_classified_as_tone4(f0_contour_factory):
    f0 = f0_contour_factory("falling")
    voiced = np.ones_like(f0, dtype=bool)
    align = [SyllableAlign(char="骂", pinyin="ma", tone=4,
                            start=0.0, end=len(f0) * 0.01)]
    score = score_tone(f0, voiced, align)
    assert score.per_syllable[0].detected == 4


def test_tone_dipping_is_classified_as_tone3(f0_contour_factory):
    f0 = f0_contour_factory("dipping")
    voiced = np.ones_like(f0, dtype=bool)
    align = [SyllableAlign(char="马", pinyin="ma", tone=3,
                            start=0.0, end=len(f0) * 0.01)]
    score = score_tone(f0, voiced, align)
    assert score.per_syllable[0].detected == 3


def test_tone_unvoiced_segment_marked_undetectable():
    f0 = np.zeros(20, dtype=np.float32)
    voiced = np.zeros_like(f0, dtype=bool)
    align = [SyllableAlign(char="是", pinyin="shi", tone=4,
                            start=0.0, end=0.2)]
    score = score_tone(f0, voiced, align)
    assert score.per_syllable[0].detected == 0
    assert score.per_syllable[0].score == 0.0


# --------------------------------------------------------------- accuracy DTW
def test_accuracy_identical_mfcc_scores_high():
    # Two identical MFCC sequences should produce ~max score.
    mfcc = np.random.default_rng(0).standard_normal((30, 13)).astype(np.float32)
    align = [SyllableAlign("你", "ni", 3, 0.0, 0.30)]
    result = score_accuracy(mfcc, mfcc.copy(), align, align)
    assert result.overall > 70.0
    assert len(result.per_syllable) == 1


def test_accuracy_unrelated_mfcc_scores_low():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((30, 13)).astype(np.float32)
    b = rng.standard_normal((30, 13)).astype(np.float32) * 5 + 10
    align = [SyllableAlign("你", "ni", 3, 0.0, 0.30)]
    result = score_accuracy(a, b, align, align)
    assert 0.0 <= result.overall <= 100.0


# ---------------------------------------------------------------- fluency
def test_fluency_handles_zero_duration():
    score = score_fluency([], [], 0.0)
    assert score.overall == 0.0


def test_fluency_normal_speech_rate_high():
    # 4 syllables / second, no pauses → near-perfect fluency.
    vad = [(0.0, 2.0)]
    align = [SyllableAlign(c, "x", 1, i * 0.25, (i + 1) * 0.25)
             for i, c in enumerate("你好世界你好世界")]
    score = score_fluency(vad, align, 2.0)
    assert score.overall > 70.0


def test_fluency_long_pauses_penalized():
    # Big silence stretch in the middle.
    vad = [(0.0, 0.5), (4.0, 4.5)]
    align = [SyllableAlign(c, "x", 1, i * 0.1, (i + 1) * 0.1)
             for i, c in enumerate("你好世界")]
    score = score_fluency(vad, align, 4.5)
    assert score.pause_count >= 1
    assert score.pause_ratio > 0.5


# ---------------------------------------------------------------- prosody
def test_prosody_identical_contours_high_similarity():
    f0 = np.linspace(150, 250, 80).astype(np.float32)
    voiced = np.ones_like(f0, dtype=bool)
    score = score_prosody(f0, voiced, f0.copy(), voiced.copy())
    assert score.contour_similarity > 50.0


def test_prosody_short_input_returns_zero():
    f0 = np.array([0.0, 0.0], dtype=np.float32)
    voiced = np.zeros_like(f0, dtype=bool)
    score = score_prosody(f0, voiced, f0, voiced)
    assert score.overall == 0.0


# ---------------------------------------------------------------- completeness
def test_completeness_perfect():
    s = score_completeness("你好世界", "你好世界")
    assert s.overall > 95.0


def test_completeness_partial_read():
    s = score_completeness("你好", "你好世界明天见")
    assert 10.0 < s.overall < 95.0


def test_completeness_empty():
    s = score_completeness("", "你好世界")
    assert s.overall == 0.0


# ---------------------------------------------------------------- aggregator
@dataclass
class _MockScore:
    overall: float
    per_syllable: list = None


def test_aggregator_combines_dimensions():
    from src.scoring.accuracy import AccuracyScore, SyllableAccuracy
    from src.scoring.tone import SyllableTone, ToneScore
    from src.scoring.fluency import FluencyScore
    from src.scoring.prosody import ProsodyScore
    from src.scoring.completeness import CompletenessScore

    acc = AccuracyScore(overall=80,
                        per_syllable=[SyllableAccuracy("你", 80, 0.2, 0.3)])
    tone = ToneScore(overall=90,
                     per_syllable=[SyllableTone("你", 3, 3, True, 1.0, 90)])
    fluency = FluencyScore(70, 4.0, 0, 0.0, 0.0)
    prosody = ProsodyScore(60, 60, 60, 8.0)
    comp = CompletenessScore(95, 0.05, 0.95)

    result = aggregate({
        "accuracy": acc, "tone": tone, "fluency": fluency,
        "prosody": prosody, "completeness": comp,
    })
    assert 50 < result.overall < 100
    assert result.dims["accuracy"] == 80
    assert len(result.per_syllable) == 1
    assert result.per_syllable[0]["char"] == "你"
