"""Unit tests for the scoring modules (M4)."""
from dataclasses import dataclass

import numpy as np

from src.asr.aligner import SyllableAlign
from src.scoring.accuracy import score_accuracy
from src.scoring.aggregator import aggregate
from src.scoring.completeness import score_completeness
from src.scoring.confidence import score_confidence
from src.scoring.fluency import score_fluency
from src.scoring.prosody import score_prosody
from src.scoring.tone import score_tone


# --------------------------------------------------------- tone classification
def _fake_f0_segment(contour, hop_sec=0.01):
    """Build a (f0_array, voiced_mask, alignment) pinned to 0..n_frames*hop."""
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


def test_tone_reference_identical_contour_scores_high(f0_contour_factory):
    f0 = f0_contour_factory("rising")
    voiced = np.ones_like(f0, dtype=bool)
    align = [SyllableAlign(char="麻", pinyin="ma", tone=2,
                            start=0.0, end=len(f0) * 0.01)]
    score = score_tone(f0, voiced, align, f0.copy(), voiced.copy(), align)
    syl = score.per_syllable[0]
    assert syl.score > 90.0
    assert syl.ref_similarity is not None and syl.ref_similarity > 90.0
    assert syl.correct


def test_tone_reference_opposite_contour_penalized(f0_contour_factory):
    user_f0 = f0_contour_factory("falling")
    ref_f0 = f0_contour_factory("rising")
    voiced = np.ones_like(user_f0, dtype=bool)
    align = [SyllableAlign(char="麻", pinyin="ma", tone=2,
                            start=0.0, end=len(user_f0) * 0.01)]
    score = score_tone(user_f0, voiced, align, ref_f0, voiced.copy(), align)
    syl = score.per_syllable[0]
    assert syl.score < 85.0
    assert syl.ref_similarity is not None
    assert syl.reason == "F0 轮廓与标准音差异较大"


# --------------------------------------------------------------- accuracy DTW
def test_accuracy_identical_mfcc_scores_high():
    # Two identical MFCC sequences should produce ~max score.
    mfcc = np.random.default_rng(0).standard_normal((30, 13)).astype(np.float32)
    align = [SyllableAlign("你", "ni", 3, 0.0, 0.30)]
    voiced = np.ones(30, dtype=bool)
    result = score_accuracy(mfcc, mfcc.copy(), align, align, voiced, voiced.copy())
    assert result.overall > 70.0
    assert len(result.per_syllable) == 1
    syl = result.per_syllable[0]
    assert syl.initial == "n"
    assert syl.final == "i"
    assert syl.initial_score is not None
    assert syl.final_score is not None
    assert syl.articulation_score == 100.0


def test_accuracy_unrelated_mfcc_scores_low():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((30, 13)).astype(np.float32)
    b = rng.standard_normal((30, 13)).astype(np.float32) * 5 + 10
    align = [SyllableAlign("你", "ni", 3, 0.0, 0.30)]
    result = score_accuracy(a, b, align, align)
    assert 0.0 <= result.overall <= 100.0


def test_accuracy_initial_and_final_segments_are_scored_separately():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((60, 13)).astype(np.float32)
    user_initial_bad = ref.copy()
    user_initial_bad[:18] = rng.standard_normal((18, 13)).astype(np.float32) * 5 + 10
    user_final_bad = ref.copy()
    user_final_bad[22:] = rng.standard_normal((38, 13)).astype(np.float32) * 5 + 10
    align = [SyllableAlign("中", "zhong", 1, 0.0, 0.60)]

    initial_result = score_accuracy(user_initial_bad, ref, align, align).per_syllable[0]
    final_result = score_accuracy(user_final_bad, ref, align, align).per_syllable[0]

    assert initial_result.initial == "zh"
    assert initial_result.final == "ong"
    assert initial_result.initial_score is not None
    assert initial_result.final_score is not None
    assert final_result.initial_score is not None
    assert final_result.final_score is not None
    assert initial_result.initial_score < initial_result.final_score
    assert final_result.final_score < final_result.initial_score


def test_accuracy_penalizes_missing_voiced_coverage():
    rng = np.random.default_rng(0)
    mfcc = rng.standard_normal((30, 13)).astype(np.float32)
    align = [SyllableAlign("你", "ni", 3, 0.0, 0.30)]
    ref_voiced = np.ones(30, dtype=bool)
    user_voiced = np.zeros(30, dtype=bool)

    result = score_accuracy(
        mfcc, mfcc.copy(), align, align,
        user_voiced=user_voiced,
        ref_voiced=ref_voiced,
    )
    syl = result.per_syllable[0]
    assert syl.voiced_coverage_score == 0.0
    assert syl.articulation_score == 0.0
    assert syl.score < 60.0


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
    assert score.overall < 65.0


def test_fluency_formal_slow_reading_is_not_over_penalized():
    # 正式示范朗读可能偏慢，但字间节奏稳定、停顿少，应保持较高流利度。
    vad = [(0.0, 2.7)]
    align = [SyllableAlign(c, "x", 1, i * 0.45, (i + 1) * 0.45)
             for i, c in enumerate("今天天气真好")]
    score = score_fluency(vad, align, 3.0)
    assert score.speech_rate < 2.5
    assert score.overall > 75.0


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
    assert [x.covered for x in s.per_syllable] == [True, True, True, True]


def test_completeness_partial_read():
    s = score_completeness("你好", "你好世界明天见")
    assert 10.0 < s.overall < 95.0
    assert [x.covered for x in s.per_syllable[:2]] == [True, True]
    assert any(not x.covered for x in s.per_syllable)


def test_completeness_marks_missing_middle_character():
    s = score_completeness("你世界", "你好世界")
    assert [x.covered for x in s.per_syllable] == [True, False, True, True]


def test_completeness_empty():
    s = score_completeness("", "你好世界")
    assert s.overall == 0.0
    assert [x.covered for x in s.per_syllable] == [False, False, False, False]


# ---------------------------------------------------------------- confidence
def test_confidence_high_when_evidence_is_complete():
    from src.scoring.accuracy import AccuracyScore, SyllableAccuracy
    from src.scoring.completeness import CompletenessScore, SyllableCompleteness
    from src.scoring.tone import SyllableTone, ToneScore

    dim_scores = {
        "accuracy": AccuracyScore(
            95,
            [SyllableAccuracy("你", 95, 0.1, 0.3, articulation_score=95)],
        ),
        "tone": ToneScore(
            96,
            [SyllableTone("你", 3, 3, True, 0.9, 96, coverage=1.0)],
        ),
        "completeness": CompletenessScore(
            100, 0.0, 1.0,
            [SyllableCompleteness("你", True, 0)],
        ),
    }

    score = score_confidence(
        dim_scores,
        user_voiced=np.ones(40, dtype=bool),
        user_duration=1.0,
        user_voiced_duration=0.8,
        has_tts_reference=True,
        has_asr=True,
    )
    assert score.overall > 90.0
    assert score.dims["asr"] == 100.0
    assert score.per_syllable[0]["level"] == "高"


def test_confidence_drops_when_reference_and_asr_are_missing():
    dim_scores = {}
    score = score_confidence(
        dim_scores,
        user_voiced=np.zeros(20, dtype=bool),
        user_duration=1.0,
        user_voiced_duration=0.1,
        has_tts_reference=False,
        has_asr=False,
    )
    assert score.overall < 60.0
    assert score.dims["reference"] == 55.0
    assert any("没有可用 TTS" in note for note in score.notes)
    assert any("没有可用 ASR" in note for note in score.notes)


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
    from src.scoring.completeness import CompletenessScore, SyllableCompleteness
    from src.scoring.confidence import ConfidenceScore

    acc = AccuracyScore(
        overall=80,
        per_syllable=[
            SyllableAccuracy(
                "你", 80, 0.2, 0.3,
                initial_score=75, final_score=85,
                articulation_score=90,
                initial="n", final="i",
            ),
        ],
    )
    tone = ToneScore(
        overall=90,
        per_syllable=[
            SyllableTone(
                "你", 2, 2, True, 1.0, 90,
                pinyin="ni", lexical_tone=3, tone_rule="三声连读",
                start=0.0, end=0.3,
            )
        ],
    )
    fluency = FluencyScore(70, 4.0, 0, 0.0, 0.0)
    prosody = ProsodyScore(60, 60, 60, 8.0)
    comp = CompletenessScore(
        95, 0.05, 0.95,
        per_syllable=[SyllableCompleteness("你", True, 0)],
    )
    confidence = ConfidenceScore(
        88,
        dims={"signal": 90, "reference": 100, "asr": 95, "f0": 80, "accuracy": 90},
        per_syllable=[{"char": "你", "score": 86, "level": "高"}],
    )

    result = aggregate({
        "accuracy": acc, "tone": tone, "fluency": fluency,
        "prosody": prosody, "completeness": comp, "confidence": confidence,
    })
    assert 50 < result.overall < 100
    assert result.dims["accuracy"] == 80
    assert len(result.per_syllable) == 1
    assert result.per_syllable[0]["char"] == "你"
    assert result.per_syllable[0]["pinyin"] == "ni"
    assert result.per_syllable[0]["lexical_tone"] == 3
    assert result.per_syllable[0]["expected_tone"] == 2
    assert result.per_syllable[0]["tone_rule"] == "三声连读"
    assert result.per_syllable[0]["tone_confidence"] == 1.0
    assert result.per_syllable[0]["initial"] == "n"
    assert result.per_syllable[0]["final"] == "i"
    assert result.per_syllable[0]["initial_score"] == 75
    assert result.per_syllable[0]["final_score"] == 85
    assert result.per_syllable[0]["articulation_score"] == 90
    assert result.per_syllable[0]["completeness_ok"]
    assert result.per_syllable[0]["confidence_level"] == "高"
    assert result.confidence["overall"] == 88
