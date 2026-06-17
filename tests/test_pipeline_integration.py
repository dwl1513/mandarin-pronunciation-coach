"""End-to-end smoke test for the assessment pipeline.

We intentionally pass `use_asr=False, use_tts_reference=False` so the test
does not require a model download or network access — this keeps the test
fast and CI-friendly while still exercising:
    preprocess → align (VAD fallback) → MFCC → F0 → all 5 scorers →
    aggregate → build_report
"""
import numpy as np

import src.pipeline as pipeline
from src.pipeline import assess


def _synthetic_utterance(sr=16000, n_syllables=4):
    """Build a fake 'utterance' = N voiced bursts separated by tiny silences."""
    rng = np.random.default_rng(0)
    chunks = []
    f0_list = [180, 230, 200, 250][:n_syllables]
    for f0 in f0_list:
        t = np.arange(int(sr * 0.20)) / sr
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
        sig = sum(
            np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)
        ) * env * 0.4
        sig += rng.normal(0, 0.002, sig.shape)
        chunks.append(sig.astype(np.float32))
        chunks.append(np.zeros(int(sr * 0.08), dtype=np.float32))
    return np.concatenate(chunks), sr


def test_pipeline_smoke_runs_end_to_end():
    wav, sr = _synthetic_utterance(n_syllables=4)
    art = assess((sr, wav), reference_text="你好朋友",
                 use_asr=False, use_tts_reference=False)

    assert isinstance(art.report, dict)
    assert "overall" in art.report
    assert "confidence" in art.report
    assert 0.0 <= art.report["overall"] <= 100.0
    assert 0.0 <= art.report["confidence"]["overall"] <= 100.0
    # ASR 关闭时跳过完整度，避免测试触发模型下载或云端调用。
    for dim in ("accuracy", "tone", "fluency", "prosody"):
        assert dim in art.report["dims"]
    assert "completeness" not in art.report["dims"]
    # Per-syllable view should match the reference text length.
    assert len(art.report["per_syllable"]) == 4
    chars = [s["char"] for s in art.report["per_syllable"]]
    assert chars == ["你", "好", "朋", "友"]
    assert "initial_score" in art.report["per_syllable"][0]
    assert "final_score" in art.report["per_syllable"][0]


def test_pipeline_report_markdown_non_empty():
    wav, sr = _synthetic_utterance(n_syllables=4)
    art = assess((sr, wav), reference_text="你好朋友",
                 use_asr=False, use_tts_reference=False)
    md = art.report["markdown"]
    assert isinstance(md, str)
    assert "总分" in md
    assert "你" in md and "好" in md
    assert "声母分" in md
    assert "韵母分" in md
    assert "完整度" in md
    assert "评分可信度" in md


def test_parse_tts_engines_supports_multi_reference():
    parsed = pipeline._parse_tts_engines("mimo-tts,aliyun-tts:Neil")
    assert parsed == [("mimo-tts", None), ("aliyun-tts", "Neil")]
