"""Quick live sanity check: TTS → preprocess → MFCC → F0 → align → score.

This actually downloads the wav2vec2 model on first run (~1.2 GB) and calls
edge-tts over the network, so it's NOT a unit test. Run manually to verify
the full online stack works in your env:

    uv run python scripts/smoke_check.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feedback.tts import synth_reference          # noqa: E402
from src.pipeline import assess                       # noqa: E402


def _primary_tts_engine(engine: str | None) -> str | None:
    if not engine:
        return None
    return engine.split(",", 1)[0].strip() or None


def main():
    parser = argparse.ArgumentParser(description="普通话发音评测完整链路烟测")
    parser.add_argument("--text", default="今天天气真好。",
                        help="用于烟测的参考文本")
    parser.add_argument("--tts-engine", default=None,
                        help=("标准参考音 TTS 引擎，例如 mimo-tts、aliyun-tts，"
                              "或 mimo-tts,aliyun-tts 启用多参考音融合"))
    parser.add_argument("--tts-voice", default=None,
                        help="TTS 音色名，例如 edge-tts 的 zh-CN-XiaoxiaoNeural 或阿里云 Cherry")
    parser.add_argument("--asr-engine", default=None,
                        choices=["wav2vec2", "aliyun-asr", "auto"],
                        help="完整度识别使用的 ASR 引擎")
    args = parser.parse_args()

    text = args.text
    print(f"[1/3] Synthesizing reference for: {text!r}")
    ref_path = synth_reference(
        text, engine=_primary_tts_engine(args.tts_engine), voice=args.tts_voice,
    )
    print(f"      → {ref_path}")

    print("[2/3] Running pipeline against the TTS audio (so user==reference).")
    art = assess(
        ref_path, text, use_asr=True, use_tts_reference=True,
        asr_engine=args.asr_engine,
        tts_engine=args.tts_engine, tts_voice=args.tts_voice,
    )
    rep = art.report
    print(f"[3/3] Done. Recognized text: {art.recognized_text!r}")
    print(f"      Total score:  {rep['overall']:.1f}")
    print(f"      Dimensions:   {rep['dims']}")
    if rep.get("confidence"):
        print(f"      Confidence:   {rep['confidence']['overall']:.1f}")
    print(f"      n syllables:  {len(rep['per_syllable'])}")
    for s in rep["per_syllable"]:
        initial = s.get("initial_score")
        final = s.get("final_score")
        articulation = s.get("articulation_score")
        initial_show = "-" if initial is None else f"{initial:.0f}"
        final_show = "-" if final is None else f"{final:.0f}"
        articulation_show = "-" if articulation is None else f"{articulation:.0f}"
        print(f"        - {s['char']}  tone {s['expected_tone']}→{s['detected_tone']}"
              f"  acc {s['acc_score']:.0f}"
              f"  initial/final {initial_show}/{final_show}"
              f"  voiced {articulation_show}"
              f"  tone {s['tone_score']:.0f}")


if __name__ == "__main__":
    main()
