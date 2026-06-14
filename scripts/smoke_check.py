"""Quick live sanity check: TTS → preprocess → MFCC → F0 → align → score.

This actually downloads the wav2vec2 model on first run (~1.2 GB) and calls
edge-tts over the network, so it's NOT a unit test. Run manually to verify
the full online stack works in your env:

    python scripts/smoke_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feedback.tts import synth_reference          # noqa: E402
from src.pipeline import assess                       # noqa: E402


def main():
    text = "今天天气真好。"
    print(f"[1/3] Synthesizing reference for: {text!r}")
    ref_path = synth_reference(text)
    print(f"      → {ref_path}")

    print(f"[2/3] Running pipeline against the TTS audio (so user==reference).")
    art = assess(ref_path, text, use_asr=True, use_tts_reference=True)
    rep = art.report
    print(f"[3/3] Done. Recognized text: {art.recognized_text!r}")
    print(f"      Total score:  {rep['overall']:.1f}")
    print(f"      Dimensions:   {rep['dims']}")
    print(f"      n syllables:  {len(rep['per_syllable'])}")
    for s in rep["per_syllable"]:
        print(f"        - {s['char']}  tone {s['expected_tone']}→{s['detected_tone']}"
              f"  acc {s['acc_score']:.0f}  tone {s['tone_score']:.0f}")


if __name__ == "__main__":
    main()
